**Project Title**: Large Language Models and OCR-Text Detection to Improve Educational Independence and Accessibility For Students With Learning Impairments 

Category: Computer Science

**Introduction**:
Many young students with impairments, both physical and cognitive, face significant challenges in education. Physical disabilities, such as visual impairments or difficulties with neural information processing, can hinder fundamental tasks like text recognition and written composition. For instance, locating specific content in a textbook or conducting textual analysis can become cognitively demanding. Neurodevelopmental disorders, such as ADHD, further complicate learning by impairing executive functions, such as attention regulation and working memory, thus hindering the development of a robust educational foundation. This research aims to address these barriers by developing an AI-powered assistive technology integrating Optical Character Recognition (OCR) and customizable Large Language Model (LLM) parameters. These systems will be tailored to individual learning profiles, enabling students to engage with educational materials through multimodal interaction effectively. It is proven that large language models have the capacity to aid in education, with studeas revealing ChatGPT’s capabilities in generating consistent answers across disciplines, balancing depth and breadth. The societal impact of this project lies in its potential to advance educational equity and accessibility by empowering young students with disabilities to achieve their academic and personal potential. Furthermore, this project aims to increase independence and self-determination, two fundamental aspects of achieving a high quality of life for people with disabilities.

**Supplies**: 
Pi Camera2 (for text detection via OCR)
Raspberry Pi 5
OpenAI API Key
Sample educational materials

**Step 1: Code**
  Imports:
    
    from openai import OpenAI
    import cv2
    import time
    from picamera2 import Picamera2, Preview
    import pytesseract
    from pytesseract import Output
    import argparse 
    
    
    client = OpenAI()
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

Using pip or alternative python package installers, install the libraries for openai, openCV, pytesseract, picamera2, and argparse on your Raspberry Pi. It is recommended that you install these libraries to a virtual environment. Then import the libraries as shown above. Note that to use the OpenAI library, you must have access to an api key which can be created on platform.openai.com. You can purchase API tokens for a relatively cheap amount depending on the model. For these purposes, GPT 4o-mini was used. Generate and store your API Key in a safe location. The last statement sets up the pretrained text detection algorithm. This model must be downloaded from the internet. 

Capturing the photo: 


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
  
Connect your PiCamera2 to your Raspberry Pi. This code initializes the camera as an object, sets up a preview configuration, and starts the preview using the QTGL renderer. It then captures a frame after a short delay, saves it as "captured_image.jpg" using OpenCV, and finally stops and closes the camera to release resources.





    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="/captured_image.jpg")
    ap.add_argument("-c", "--min-conf", type=int, default=0,
     help="minimum confidence value to filter weak text detection")
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


       # bounding box
         text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
           1.2, (0, 0, 255), 3)
           
    # output image
    cv2.imshow("Image", image)
    print("")
    print("Final: ",final)
  
  
    cv2.destroyAllWindows()



This Python performs the Optical Character Recognition (OCR) on the captured image. Once the image is loaded, it is converted from BGR to RGB color format, as Tesseract expects RGB input. Tesseract’s image_to_data() function is then used to extract detailed information about any text it detects, including the bounding box coordinates, the text string, and a confidence score for each detection. The script then iterates through each detected text item, checking if its confidence score exceeds the user-defined threshold. If it does, the word is printed to the console, added to a final string that accumulates all valid text, and highlighted in the original image with a green rectangle and red text using OpenCV’s drawing functions. After all the text has been processed, the script displays the annotated image in a window using cv2.imshow() and prints out the full extracted text. This provides both a visual and textual summary of the OCR results. The script ends with cv2.destroyAllWindows() to properly close the display window. 



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
           {"role": "system", "content": "You are a tutor. Your goal is to best assist the user, in a concise way, that helps them with their homework. This student likely has a learning impairment. Assume they are a 1st grader with some form of an attention deficit.             Provide concise answers, be patient, and use words in an understandable manner. Also do not use latex, use symbols like '/' and '*' instead. You will be inputted with text following this blurb as well, text that is scanned by the user and sent to you.                 Make sure to account for errors in text detection. You are receiving the prompt now: \n" + final},
           {"role": "user", "content": user_input}
       ]
     )


     print(completion.choices[0].message.content)

This portion of the code loads the text information from the image into a gpt 4o mini prompt alongside contextual information that the chatbot can use. This context can be altered according to the needs of the individual. Until the user decides to quit, they can interact with the chatbot continually. 

For Use: To use the device, you need to set your stored API key. For linux, you set the environmental variable in your terminal beforehand using the following command: 
export OPENAI_API_KEY="your-api-key-here"



