from generate_files import generate_files
import config
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    seq_files = args if args else config.GetSeqFiles()
    generate_files(config.ACTUAL_FOLDER, seq_files, description="actual data")