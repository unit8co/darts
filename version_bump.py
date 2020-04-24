### INSTRUCTIONS ###

# To bump the current version of u8timeseries, please execute the version bumping script from the root:
#
# >>> python version_bump.py
# 
# The user will then be prompted to enter the desired bump settings.
# Alternatively, the settings can also be passed as arguments when calling the script:
#
# >>> python version_bump.py -b BUMP -r RELEASE -p PATH
#
# BUMP should be an integer between 0 and 3 indicating the type of version bump.
# (0 for no version increment, 1 for Major, 2 for Minor, 3 for Patch)
# RELEASE is expected to be 'y' or 'n', indicating whether or not the SNAPSHOT suffix should be added.
# If either of the arguments is missing, or if an invalid value is given, the user will be prompted
# to enter them again.
# PATH should be the path to the version file that needs to be bumped.
#
# The versioning of u8timeseries follows the semantic versioning specification (https://semver.org).

import argparse

VERSION_FILE= "u8timeseries/VERSION"

# set up arguments
parser = argparse.ArgumentParser(prog="version bump")
parser.add_argument('-b', '--bump', help="BUMP should be an integer between 0 and 3" \
                                         " indicating the type of version bump.")
parser.add_argument('-r', '--release', help="RELEASE is expected to be 'y' or 'n', indicating" \
                                            " whether or not the SNAPSHOT suffix should be added.")
parser.add_argument('-p', '--path', help="PATH is expected to be the path to the VERSION file.")
args = parser.parse_args()

# retrieve current version
if args.path != None: VERSION_FILE = args.path
with open(VERSION_FILE, "r") as f:
    version_string = f.read()
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
    print("Is this an official release? A SNAPSHOT tag will be added if not. (y/n)")
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
with open(VERSION_FILE, "w") as f:
    f.write(new_version_string)
print("The current version has been updated to", new_version_string)