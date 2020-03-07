import os
import time
import schedule


def update():
    os.system("dataAPI.py")


schedule.every().wednesday.at("13:15").do(update)

while 1:
    schedule.run_pending()
    time.sleep(1)
