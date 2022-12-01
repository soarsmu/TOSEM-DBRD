# https://github.com/bugzilla/bugzilla/blob/5.2/Bugzilla/Bug.pm

sub possible_duplicates {
  my ($class, $params) = @_;
  my $short_desc = $params->{summary};
  my $products   = $params->{products} || [];
  my $limit      = $params->{limit} || MAX_POSSIBLE_DUPLICATES;
  $limit    = MAX_POSSIBLE_DUPLICATES if $limit > MAX_POSSIBLE_DUPLICATES;
  $products = [$products]             if !ref($products) eq 'ARRAY';

  my $orig_limit = $limit;
  detaint_natural($limit)
    || ThrowCodeError('param_must_be_numeric',
    {function => 'possible_duplicates', param => $orig_limit});

  my $dbh   = Bugzilla->dbh;
  my $user  = Bugzilla->user;
  my @words = split(/[\b\s]+/, $short_desc || '');

  # Remove leading/trailing punctuation from words
  foreach my $word (@words) {
    $word =~ s/(?:^\W+|\W+$)//g;
  }

  # And make sure that each word is longer than 2 characters.
  @words = grep { defined $_ and length($_) > 2 } @words;

  return [] if !@words;

  my ($where_sql, $relevance_sql);
  if ($dbh->FULLTEXT_OR) {
    my $joined_terms = join($dbh->FULLTEXT_OR, @words);
    ($where_sql, $relevance_sql)
      = $dbh->sql_fulltext_search('bugs_fulltext.short_desc', $joined_terms);
    $relevance_sql ||= $where_sql;
  }
  else {
    my (@where, @relevance);
    foreach my $word (@words) {
      my ($term, $rel_term)
        = $dbh->sql_fulltext_search('bugs_fulltext.short_desc', $word);
      push(@where,     $term);
      push(@relevance, $rel_term || $term);
    }

    $where_sql     = join(' OR ', @where);
    $relevance_sql = join(' + ',  @relevance);
  }

  my $product_ids = join(',', map { $_->id } @$products);
  my $product_sql = $product_ids ? "AND product_id IN ($product_ids)" : "";

  # Because we collapse duplicates, we want to get slightly more bugs
  # than were actually asked for.
  my $sql_limit = $limit + 5;

  my $possible_dupes = $dbh->selectall_arrayref(
    "SELECT bugs.bug_id AS bug_id, bugs.resolution AS resolution,
                ($relevance_sql) AS relevance
           FROM bugs
                INNER JOIN bugs_fulltext ON bugs.bug_id = bugs_fulltext.bug_id
          WHERE ($where_sql) $product_sql
       ORDER BY relevance DESC, bug_id DESC " . $dbh->sql_limit($sql_limit),
    {Slice => {}}
  );

  my @actual_dupe_ids;

  # Resolve duplicates into their ultimate target duplicates.
  foreach my $bug (@$possible_dupes) {
    my $push_id = $bug->{bug_id};
    if ($bug->{resolution} && $bug->{resolution} eq 'DUPLICATE') {
      $push_id = _resolve_ultimate_dup_id($bug->{bug_id});
    }
    push(@actual_dupe_ids, $push_id);
  }
  @actual_dupe_ids = uniq @actual_dupe_ids;
  if (scalar @actual_dupe_ids > $limit) {
    @actual_dupe_ids = @actual_dupe_ids[0 .. ($limit - 1)];
  }

  my $visible = $user->visible_bugs(\@actual_dupe_ids);
  return $class->new_from_list($visible);
}

# https://github.com/bugzilla/bugzilla/blob/5.2/Bugzilla/DB.pm
sub sql_fulltext_search {
  my ($self, $column, $text) = @_;

  # This is as close as we can get to doing full text search using
  # standard ANSI SQL, without real full text search support. DB specific
  # modules should override this, as this will be always much slower.

  # make the string lowercase to do case insensitive search
  my $lower_text = lc($text);

  # split the text we're searching for into separate words. As a hack
  # to allow quicksearch to work, if the field starts and ends with
  # a double-quote, then we don't split it into words. We can't use
  # Text::ParseWords here because it gets very confused by unbalanced
  # quotes, which breaks searches like "don't try this" (because of the
  # unbalanced single-quote in "don't").
  my @words;
  if ($lower_text =~ /^"/ and $lower_text =~ /"$/) {
    $lower_text =~ s/^"//;
    $lower_text =~ s/"$//;
    @words = ($lower_text);
  }
  else {
    @words = split(/\s+/, $lower_text);
  }

  # surround the words with wildcards and SQL quotes so we can use them
  # in LIKE search clauses
  @words = map($self->quote("\%$_\%"), @words);

  # untaint words, since they are safe to use now that we've quoted them
  trick_taint($_) foreach @words;

  # turn the words into a set of LIKE search clauses
  @words = map("LOWER($column) LIKE $_", @words);

  # search for occurrences of all specified words in the column
  return join(" AND ", @words),
    "CASE WHEN (" . join(" AND ", @words) . ") THEN 1 ELSE 0 END";
}

sub selectall_arrayref {
  my $self     = shift;
  my $stmt     = shift;
  my $new_stmt = (ref $stmt) ? $stmt : adjust_statement($stmt);
  unshift @_, $new_stmt;
  my $ref = $self->SUPER::selectall_arrayref(@_);
  return undef if !defined $ref;

  foreach my $row (@$ref) {
    if (ref($row) eq 'ARRAY') {
      _fix_arrayref($row);
    }
    elsif (ref($row) eq 'HASH') {
      _fix_hashref($row);
    }
  }

  return $ref;
}