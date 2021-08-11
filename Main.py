from speech_to_text import STT
from speech import speak
from id_create import Predict_image
stt = STT()



def id_creater():
    try:

      Predict_image()

    except:
        speak("Sorry somthing went wrong")

