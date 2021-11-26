"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def download_from_google_drive(file_id: str, file_name: str, path: str):
    if not (isinstance(file_id, str) and isinstance(file_name, str) and isinstance(path, str)):
        raise TypeError

    file_path = os.path.join(path, file_name)

    if os.path.isfile(file_path):
        print(' pretrained model "%s" is already exist in "%s"' (file_name, path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        download_command = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
            FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=path)
        print('dowloading pretrained model "%s" ...' % (file_name))
        os.system(download_command)

#
# from __future__ import print_function
# import pickle
# import os.path
# import io
# import shutil
# import requests
# from mimetypes import MimeTypes
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
#
#
# class DriveAPI:
#     global SCOPES
#
#     # Define the scopes
#     SCOPES = ['https://www.googleapis.com/auth/drive']
#
#     def __init__(self):
#
#         # Variable self.creds will
#         # store the user access token.
#         # If no valid token found
#         # we will create one.
#         self.creds = None
#
#         # The file token.pickle stores the
#         # user's access and refresh tokens. It is
#         # created automatically when the authorization
#         # flow completes for the first time.
#
#         # Check if file token.pickle exists
#         if os.path.exists('token.pickle'):
#             # Read the token from the file and
#             # store it in the variable self.creds
#             with open('token.pickle', 'rb') as token:
#                 self.creds = pickle.load(token)
#
#         # If no valid credentials are available,
#         # request the user to log in.
#         if not self.creds or not self.creds.valid:
#
#             # If token is expired, it will be refreshed,
#             # else, we will request a new one.
#             if self.creds and self.creds.expired and self.creds.refresh_token:
#                 self.creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     'credentials.json', SCOPES)
#                 self.creds = flow.run_local_server(port=0)
#
#             # Save the access token in token.pickle
#             # file for future usage
#             with open('token.pickle', 'wb') as token:
#                 pickle.dump(self.creds, token)
#
#         # Connect to the API service
#         self.service = build('drive', 'v3', credentials=self.creds)
#
#         # request a list of first N files or
#         # folders with name and id from the API.
#         results = self.service.files().list(
#             pageSize=100, fields="files(id, name)").execute()
#         items = results.get('files', [])
#
#         # print a list of files
#
#         print("Here's a list of files: \n")
#         print(*items, sep="\n", end="\n\n")
#
#     def FileDownload(self, file_id, file_name):
#         request = self.service.files().get_media(fileId=file_id)
#         fh = io.BytesIO()
#
#         # Initialise a downloader object to download the file
#         downloader = MediaIoBaseDownload(fh, request, chunksize=204800)
#         done = False
#
#         try:
#             # Download the data in chunks
#             while not done:
#                 status, done = downloader.next_chunk()
#
#             fh.seek(0)
#
#             # Write the received data to the file
#             with open(file_name, 'wb') as f:
#                 shutil.copyfileobj(fh, f)
#
#             print("File Downloaded")
#             # Return True if file Downloaded successfully
#             return True
#         except:
#
#             # Return False if something went wrong
#             print("Something went wrong.")
#             return False
#
#     def FileUpload(self, filepath):
#
#         # Extract the file name out of the file path
#         name = filepath.split('/')[-1]
#
#         # Find the MimeType of the file
#         mimetype = MimeTypes().guess_type(name)[0]
#
#         # create file metadata
#         file_metadata = {'name': name}
#
#         try:
#             media = MediaFileUpload(filepath, mimetype=mimetype)
#
#             # Create a new file in the Drive storage
#             file = self.service.files().create(
#                 body=file_metadata, media_body=media, fields='id').execute()
#
#             print("File Uploaded.")
#
#         except:
#
#             # Raise UploadError if file is not uploaded.
#             raise UploadError("Can't Upload File.")
#
#
# if __name__ == "__main__":
#     obj = DriveAPI()
#     i = int(input("Enter your choice:
#                   "1 - Download file, 2- Upload File, 3- Exit.\n"))
#
#     if i == 1:
#         f_id = input("Enter file id: ")
#         f_name = input("Enter file name: ")
#         obj.FileDownload(f_id, f_name)
#
#     elif i == 2:
#         f_path = input("Enter full file path: ")
#         obj.FileUpload(f_path)
#
#     else:
#         exit()