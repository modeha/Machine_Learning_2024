import sys

from gtts import gTTS
import os

def download_pronunciation(word_or_file, output_dir):
    """
    Downloads pronunciation as MP3 for a single word or words from a .txt file.

    Args:
        word_or_file (str): A single word or path to a .txt file containing words (one per line).
        output_dir (str): Directory where MP3 files will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isfile(word_or_file):  # If input is a file
        with open(word_or_file, 'r', encoding='utf-8') as file:
            words = file.readlines()
        words = [word.strip() for word in words if word.strip()]  # Clean words
    else:  # Assume input is a single word
        words = [word_or_file]

    for word in words:
        try:
            tts = gTTS(text=word, lang='en')  # Generate speech
            output_path = os.path.join(output_dir, f"{word}.mp3")
            tts.save(output_path)  # Save MP3 file
            print(f"Pronunciation saved for '{word}' at: {output_path}")
        except Exception as e:
            print(f"Error generating pronunciation for '{word}': {e}")

# Main function to handle both options
if __name__ == "__main__":
    output_dir = "/Users/mohsen/PycharmProjects/pronunciation/pronunciations" # Directory to save MP3 files
    # Check if an argument is provided in the terminal
    if len(sys.argv) > 1:
        input_value = sys.argv[1]  # Get the first argument passed from the terminal
    else:
        # Default value when no argument is provided
        input_value = "water"  # Change this to your default word if desired

    download_pronunciation(input_value, output_dir)