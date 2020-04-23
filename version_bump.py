import argparse

# set up arguments
parser = argparse.ArgumentParser(prog="version bump")
parser.add_argument('-b', '--bump', default="", help="BUMP should be an integer between 0 and 3" \
                                                     " indicating the type of version bump.")
parser.add_argument('-r', '--release', default="", help="RELEASE is expected to be 'y' or 'n', indicating" \
                                                        " whether or not the SNAPSHOT suffix should be added.")
args = parser.parse_args()

# retrieve current version
version_string = open("u8timeseries/VERSION", "r").read()
version_parsed = version_string.split("-")[0].split(".")
print('The current version is', version_string)

# get the desired type of version bump from user (if no valid argument has been passed)
bump_type = args.bump
while (not bump_type in ["0", "1", "2", "3"]):
    print("Enter the type of version bump: 0 (No increment), 1 (Major), 2 (Minor), 3 (Patch)")
    bump_type = input()
bump_type = int(bump_type)

# get information on whether SNAPSHOT should be added (if no valid argument has been passed)
answer = args.release
while (not answer in ["y", "n"]):
    print("Is this an official release? (y/n)")
    answer = input()
if (answer == 'y'):
    suffix = ""
else:
    suffix = "-SNAPSHOT"

# update version
if (bump_type != 0):
    relevant_part = int(version_parsed[bump_type - 1])
    version_parsed[bump_type - 1] = str(relevant_part + 1)
new_version_string = ".".join(version_parsed) + suffix

# write new version to VERSION file
version_file = open("u8timeseries/VERSION", "w")
version_file.write(new_version_string)
version_file.close()
print("The current version has been updated to", new_version_string)