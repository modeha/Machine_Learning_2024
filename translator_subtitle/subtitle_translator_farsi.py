import pysrt
from googletrans import Translator

# Load the SRT file
file_path = '/Users/mohsen/PycharmProjects/translator_subtitle/Old.2021.1080p.WEBRip.x264.AAC5.1-[YTS.MX].srt'
subtitles = pysrt.open(file_path)

# Initialize the translator
translator = Translator()

# Translate subtitles
for subtitle in subtitles:
    subtitle.text = translator.translate(subtitle.text, src='en', dest='fa').text

# Save translated file
translated_file_path = 'Translated_File.srt'
subtitles.save(translated_file_path, encoding='utf-8')

print(f"Translated file saved to: {translated_file_path}")
