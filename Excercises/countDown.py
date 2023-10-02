import time

count = 0
counts=[]
while True:
    print(count)
    counts.append(count)
    count += 1
    time.sleep(1) # wait for 1 second before continuing to the next iteration
    if input("Press 's' to stop: ").lower() == 's':
        break # stop the loop when the user enters 's'
print(counts)
