import os
import sys
from typing import List


# TODO: add a group argument to be able to select subset of notebooks
def collect_files(directory: str, file_extension: str) -> List(str):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            matching_files.append(filename)
    return matching_files


def run_examples(examples: List[str]) -> None:
    for notebook in examples:
        os.system(f"./gradlew checkExample -PexampleName={notebook}")


def main():
    path = os.environ["INPUT_PATH"]
    extension = os.environ["INPUT_TYPE"]

    collected_files = collect_files(path, extension)

    run_examples(collected_files)

    sys.exit(0)


if __name__ == "__main__":
    main()
