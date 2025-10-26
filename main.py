from openai import OpenAI

import cv2

import time

from picamera2 import Picamera2, Preview

import pytesseract

from pytesseract import Output

import argparse


client = OpenAI()


net = cv2.dnn.readNet('frozen_east_text_detection.pb')


picam2 = Picamera2()


camera_config = picam2.create_preview_configuration()

picam2.configure(camera_config)


picam2.start_preview(Preview.QTGL)

picam2.start()


time.sleep(2)


frame = picam2.capture_array()

cv2.imwrite("captured_image.jpg", frame)


picam2.stop()

picam2.close()


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", help="/captured_image.jpg")

ap.add_argument("-c", "--min-conf", type=int, default=0,

   help="mininum confidence value to filter weak text detection")

args = vars(ap.parse_args())



image = cv2.imread("captured_image.jpg")


rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pytesseract.image_to_data(rgb, output_type=Output.DICT)


final = ""


for i in range(0, len(results["text"])):

   x = results["left"][i]

   y = results["top"][i]

   w = results["width"][i]

   h = results["height"][i]


   text = results["text"][i]

   conf = int(results["conf"][i])

 

   if conf > args["min_conf"]:

       print("Confidence: {}".format(conf))

       print("Text: {}".format(text))

       print("")

       final += " " + text.strip()


       # strip out non-ASCII text so we can draw the text on the image

       # using OpenCV, then draw a bounding box around the text along

       # with the text itself

       text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

       cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,

           1.2, (0, 0, 255), 3)

# show the output image

cv2.imshow("Image", image)

print("")

print("Final: ",final)


cv2.destroyAllWindows()



import openai


client = openai.Client()


while True:

   user_input = input("Enter your question (or type 'quit' to exit): ")

   if user_input.lower() == 'quit':

       print("Goodbye!")

       break


   completion = client.chat.completions.create(

       model="gpt-4o-mini",

       messages=[

           {"role": "system", "content": "You are a tutor. You're goal is to best assist the user, in a consise way, that helps them with a homework. This student likely has a learning impairment. Assume they are a 1st grader with some form of an attention deficit. Provide consise answers, be patient, and use words in an understandable manner. Also do not use latex, use symbols like '/' and '*' instead. You will be inputted with text following this blurb as well, text that is scanned by the user and sent to you. Me sure to account for errors in text detection. You are recieving the prompt now: \n" + final},

           {"role": "user", "content": user_input}

       ]

   )


   print(completion.choices[0].message.content)
