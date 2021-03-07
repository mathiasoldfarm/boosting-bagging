def log(message):
  print(message)
  with open("logger.txt", "a") as f:
    f.write(f"{message}\n")

def reset_log():
   open("logger.txt", "w")