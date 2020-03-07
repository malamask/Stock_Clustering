import schedule
import os
import time

def update():
    os.system("dataAPI.py")

schedule.every(10).wednesday.at("12:00").do(update)
while 1:
    schedule.run_pending()
    time.sleep(1)