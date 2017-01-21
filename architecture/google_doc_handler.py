# Instructions for downloading files using these functions:
# 1. Turn on the drive API: follow through "Step 1" from
#    https://developers.google.com/drive/v3/web/quickstart/python
#    At the end, you will be instructed to download a json file and
#    name it "client_secret.json" - place this json file inside
#    the same folder as your python script
# 2. add the line from "google_doc_handler import download_file"
# 3. call download_file()
# 4. After your first time running download_file(), you may
#    remove the "client_secret.json" file from you folder

import os
import httplib2

from oauth2client.file import Storage
from apiclient.discovery import build
from oauth2client import client
from oauth2client import tools

import io
from apiclient.http import MediaIoBaseDownload

from apiclient import errors

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

CLIENT_SECRET_FILE = 'client_secret.json'
OAUTH_SCOPE = 'https://www.googleapis.com/auth/drive.readonly'
CREDENTIAL_DIR_MAME = '.credentials'
CREDS_FILE_NAME = 'ece496_emg_project.json'
APPLICATION_NAME = 'ECE496 EMG Project'

def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, CREDENTIAL_DIR_MAME)
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, CREDS_FILE_NAME)

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, OAUTH_SCOPE)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def list_files(service):
    page_token = None
    while True:
        param = {}
        if page_token:
            param['pageToken'] = page_token
        files = service.files().list(**param).execute()
        for item in files.get('files'):
            yield item
        page_token = files.get('nextPageToken')
        if not page_token:
            break

def download_file(file_name, output_dir='.'):
    downloaded_file_path = None

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    drive_service = build('drive', 'v3', http=http)

    out_path = os.path.join(os.path.dirname(__file__), output_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for item in list_files(drive_service):
        if(item.get('name') == file_name):
            _downloaded_file_path = out_path+'/'+file_name

            # If file doesn't exist, download it
            if not os.path.exists(out_path):
                print('"' + file_name + '" found, beginning download')
                fh = io.FileIO(_downloaded_file_path, 'wb')
                file_id = item.get('id')
                request = drive_service.files().get_media(fileId=file_id)
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print("Download %d%%." % int(status.progress() * 100))

            # If file already exists, ignore downlaod
            else:
                print('"' + file_name + '" already exists, skipping download')
            downloaded_file_path = _downloaded_file_path
    return downloaded_file_path