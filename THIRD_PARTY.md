# Third party software and models

lecturize is MIT licensed. At run time it downloads and uses:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and the Whisper
  models converted to CTranslate2 by SYSTRAN (MIT), derived from
  [OpenAI Whisper](https://github.com/openai/whisper) (MIT);
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT),
  [sentencepiece](https://github.com/google/sentencepiece) (Apache 2.0),
  [PyAV](https://github.com/PyAV-Org/PyAV) (BSD) with its bundled FFmpeg (LGPL);
- translation packages from the [Argos Translate](https://www.argosopentech.com/)
  index, built from [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) models by
  the University of Helsinki, released under CC-BY 4.0. If you redistribute
  output produced with them, keep the attribution: "Translated with OPUS-MT
  models (Helsinki-NLP, CC-BY 4.0) packaged by Argos Translate";
- the [Silero VAD](https://github.com/snakers4/silero-vad) model shipped inside
  faster-whisper (MIT).

The sample recording in `tests/data/lecture.wav` is synthetic speech generated
with the Windows speech synthesizer from a text I wrote; it carries no third
party rights.
