import streamlit as st
from gtts import gTTS
import os
import tempfile
from googletrans import Translator


# Function to download pronunciation
def download_pronunciation(word, output_dir, language):
    """
    Downloads pronunciation as MP3 for a given word.

    Args:
        word (str): A single word or text.
        output_dir (str): Directory where MP3 files will be saved.
        language (str): Language for text-to-speech.

    Returns:
        str: Path to the saved MP3 file.
    """
    try:
        tts = gTTS(text=word, lang=language)
        sanitized_word = "".join(c if c.isalnum() else "_" for c in word)  # Sanitize filename
        output_path = os.path.join(output_dir, f"{sanitized_word}_{language}.mp3")
        tts.save(output_path)  # Save MP3 file
        return output_path
    except Exception as e:
        st.error(f"Error generating pronunciation for '{word}': {e}")
        return None


# Streamlit app
def main():
    st.title("Pronunciation and Meaning Downloader")

    # Language selection
    st.write("Choose your preferred language:")
    language_choice = st.radio("Select Language:", ("English", "French"))

    # Input area
    if language_choice == "English":
        st.write("Enter an English word to find its meaning in French and download MP3 pronunciations for both.")
        input_label = "Enter a word in English:"
        source_lang = "en"
        target_lang = "fr"
    else:
        st.write("Enter a French word to find its meaning in English and download MP3 pronunciations for both.")
        input_label = "Enter a word in French:"
        source_lang = "fr"
        target_lang = "en"

    word = st.text_input(input_label)

    # Use a temporary directory for output
    output_dir = tempfile.mkdtemp()

    # Initialize the translator
    translator = Translator()

    if st.button("Find Meaning and Generate Pronunciations"):
        if word:
            # Translate the word
            try:
                translation = translator.translate(word, src=source_lang, dest=target_lang).text
                st.write(f"**Meaning in {target_lang.capitalize()}:** {translation}")

                # Generate pronunciation for both languages
                st.write("Generating pronunciations...")
                source_pronunciation = download_pronunciation(word, output_dir, source_lang)
                target_pronunciation = download_pronunciation(translation, output_dir, target_lang)

                # Display and allow download of the generated MP3 files
                if source_pronunciation:
                    st.audio(source_pronunciation, format="audio/mp3", start_time=0)
                    st.download_button(
                        label=f"Download Pronunciation ({language_choice}: {word})",
                        data=open(source_pronunciation, "rb").read(),
                        file_name=os.path.basename(source_pronunciation),
                        mime="audio/mp3",
                    )

                if target_pronunciation:
                    st.audio(target_pronunciation, format="audio/mp3", start_time=0)
                    st.download_button(
                        label=f"Download Pronunciation ({'French' if language_choice == 'English' else 'English'}: {translation})",
                        data=open(target_pronunciation, "rb").read(),
                        file_name=os.path.basename(target_pronunciation),
                        mime="audio/mp3",
                    )
            except Exception as e:
                st.error(f"Error finding meaning or generating pronunciations: {e}")
        else:
            st.warning("Please enter a word.")


if __name__ == "__main__":
    main()
