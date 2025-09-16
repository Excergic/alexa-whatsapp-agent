from datetime import datetime

current_datetime = datetime.now()
print(current_datetime)
current_time = current_datetime.time()
current_day = current_datetime.weekday()

print(current_time)
print(current_day)