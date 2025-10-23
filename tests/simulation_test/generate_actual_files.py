from generate_files import generate_files
import config

if __name__ == "__main__":
    seq_files = config.GetSeqFiles()
    generate_files(config.ACTUAL_FOLDER, seq_files, description="actual data")