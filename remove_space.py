import os
import sys

# for detection-results 6 <class_name> <confidence> <left> <top> <right> <bottom>
# for ground-truth 5 <class_name> <left> <top> <right> <bottom>
EXPECTED_COLS = 5

# check if a target folder has been provided
if len(sys.argv) == 1:
    print("No folder given")
    quit()

else:
    # get all files in the folder
    file_list = os.listdir(sys.argv[1])

    for filename in file_list:
        # path to the file
        path = os.path.join(sys.argv[1], filename)

        # save content of the file
        new_file = []

        with open(path, "r+") as f:
            for line in f:

                # empty file
                if line is None:
                    break

                if len(line.split()) == EXPECTED_COLS + 1:
                    # remove extra space
                    line = line.replace(" ", "", 1)

                new_file.append(line)

        # replace the contents of the file with corrected values
        with open(path, "w+") as f:
            for i in new_file:
                f.write(i)
