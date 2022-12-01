"""
First check whether the changes in GitHub is frequent
"""

from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os, time
from selenium.common.exceptions import NoSuchElementException
import json

kibana_base_url = "https://github.com/elastic/kibana/issues/"
kibana_file = '../../SABD/dataset/kibana/kibana.json'
kibana_htmls = '../data/kibana_htmls/'
kibana_edited_ids = './kibana_edited_issues.txt'

vscode_base_url = "https://github.com/microsoft/vscode/issues/"
vscode_file = '../../SABD/dataset/vscode/vscode.json'
vscode_htmls = '../data/vscode_htmls/'
vscode_edited_ids = './vscode_edited_issues.txt'

def is_edited(html_folder, file):
    # page = requests.get(url)
    with open(html_folder + file) as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    h3 = soup.find_all('h3', class_='timeline-comment-header-text')
    edited = h3[0].find_all(class_='js-comment-edit-history')
    if len(edited) > 0:
        if 'edited' in str(edited[0]):
            return True
        return False
    
def count_edits(reponame, html_folder):
    count_edits = 0
    edited_issues = list()
    for file in tqdm(os.listdir(html_folder)):
        if is_edited(html_folder, file):
            edited_issues.append(file.split('.')[0])
            count_edits += 1
    print(edited_issues)
    print(count_edits)
    with open('{}_edited_issues.txt'.format(reponame), 'w') as f:
        for issue in edited_issues:
            f.write(issue + '\n')

def download_initial_edit(repo, base_url, issue_id):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    option = webdriver.ChromeOptions()
    option.add_argument("--incognito")
    option.add_argument("--headless")
    
    driver.get('{}/{}'.format(base_url, issue_id))
    try:
        history = driver.find_element(by=By.CLASS_NAME, value='js-comment-edit-history')
        history.click()
    except NoSuchElementException:
        return
    
    # ul[class='toc chapters']
    edits = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'li[class="border-bottom css-truncate"]')))
    edits.click()
    # edit_times = driver.find_element(By.CSS_SELECTOR, ".dropdown-header.px-3.py-2.border-bottom")
    # num_edit_times = int(edit_times.text.split()[1])
    try:
        last_edit = driver.find_elements(By.CSS_SELECTOR, ".btn-link.dropdown-item.p-2")[-1]
    except IndexError:
        return
    # webdriver.ActionChains(driver).move_to_element(last_edit).click(last_edit).perform()
    driver.execute_script("arguments[0].click();", last_edit)
    WebDriverWait(driver, 20).until(lambda driver: driver.execute_script('return document.readyState') == 'complete')

    time.sleep(1)
    os.makedirs('{}'.format(repo), exist_ok=True)
    with open('{}/{}.html'.format(repo, issue_id), 'w') as f:
        f.write(driver.page_source)
        
    # WebDriverWait(driver, 20).until(EC.element_to_be_clickable(last_edit)).click()
    # init_edit = driver.find_elements(By.CSS_SELECTOR, ".btn-link.dropdown-item.p-2")[-1].click()
    # WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='markdown-body entry-content comment-body p-0']")))

    # real_edits = driver.find_elements(By.CSS_SELECTOR, ".rich-diff-level-zero")[-1]
    # print(WebDriverWait(driver, 10).until(EC.visibility_of_element_located((real_edits))).text)

if __name__ == "__main__":
    with open(kibana_edited_ids) as f:
        lines = f.readlines()
    edited_ids = set()
    for line in lines:
        edited_ids.add(line.strip())
        
    with open(kibana_file) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        cur_br = json.loads(line)  
        if not cur_br['bug_id'] in edited_ids:
            continue  
        download_initial_edit('kibana', 'https://github.com/elastic/kibana/issues/', '{}'.format(cur_br['bug_id']))
    # count_edits('kibana')
    # count_edits('vscode', vscode_htmls)