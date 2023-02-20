import os
from random import randint

for i in range(730, 2065):  # Loop over 365 days (1 year)
    for j in range(0, randint(1, 8)):  # Random commits per day
        d = str(i) + ' days ago'  # Date format
        with open('file_new.txt', 'a') as file:  # Open file.txt in append mode
            file.write(d + '\n')  # Write the date
        os.system('git add .')  # Stage the changes
        os.system(f'git commit --date="{d}" -m "commit"')  # Commit with a past date

os.system('git push -u origin main')  # Push commits to GitHub
print("Code Successfully Completed")
