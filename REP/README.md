# fast-dbrd-modified

We made minor modifications to the fast-dbrd code in order to compare REP and BM25F_ext to SABD (Soft Alignment Model for Bug Deduplication).

The modifications are  summarized as follows:
* The seeds are randomly chosen.
* Print in the log when a duplicate bug report was not reached in a specifin time window.
* We train the model even though all the reports are in the training dataset.


Original code can be found [here](https://chengniansun.bitbucket.io/projects/bug-report/fast-dbrd.tgz).


*Example to run fast-dbrd:*

	/build/bin/fast-dbrd -n rep_mozilla_2001-2009-2010_test_1 -r HOME/fast-dbrd/ranknet-configs/full-textual-no-version.cfg --ts DATA_DIR/sun_2011/mozilla_2001-2009_2010/timestamp_file.txt --time-constraint 1095 --training-duplicates 128630 --recommend DATA_DIR/sun_2011/mozilla_2001-2009_2010/dbrd_test.txt
