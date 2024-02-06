from django.shortcuts import render,HttpResponse,redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from .models import ForwardedData
from .models import EmotionData
from Evaluator.models import Notification
from MyAdmin.models import User
from django.shortcuts import render
import cv2
from django.contrib.auth.decorators import login_required
import math
import mediapipe as mp
from django.http import JsonResponse

import cv2


def Dashboard(request):
    if request.user.is_anonymous:
        return redirect("/evaluator")
    return render(request,'dashboard.html')
    


def evaluatorLogin(request):
    if request.method =="POST":
        username = request.POST.get('username')
        password =request.POST.get('password')
        
        user= authenticate(request,username=username, password=password )
        if user is not None:
            login(request,user)
            return redirect('/evaluator/dashboard')
        else:
            messages.error(request, 'Invalid Credentials!!')
            messages.error(request, 'Please check your username or password')
            return render(request,'evaluatorLogin.html')
    return render(request,'evaluatorLogin.html')

def GazeData(request):
    return render(request, 'gazedata.html')
    
def logoutUser(request):
    logout(request)
    return render(request,'evaluatorLogin.html')


def showprofile(request):
    return render(request, 'user.html')

def showProject(request):
    return render(request, 'forwarded_request.html')
def feedback(request):
    return render(request, 'feedback.html')
def cloud(request):
    return render(request, 'cloud.html')
def StartProject(request):
    return render(request,'gazeData.html')


def view_forwarded_request(request):
    # Assuming you have user authentication and evaluator is the logged-in user
    evaluator = request.user
    forwarded_requests = ForwardedData.objects.filter(assigned_evaluator=evaluator)

    return render(request, 'newProject.html', {'forwarded_requests': forwarded_requests})


def view_notifications(request):
    # Retrieve notifications for the current user
    notifications = Notification.objects.filter(recipient=request.user).order_by('-timestamp')
    return render(request, 'notifications.html', {'notifications': notifications})




















from django.views.decorators.csrf import csrf_exempt # used for views that receive data from external sources, such as POST requests from webhooks.
from . tasks import process_gaze_data # Importing the Celery task responsible for processing gaze data asynchronously.
from . models import GazeData 
import matplotlib.pyplot as plt  # creating visualizations, particularly heatmaps.
import seaborn as sns # It enhances the aesthetics and readability of Matplotlib plots.
import pandas as pd #convert database query results to a DataFrame.
from io import BytesIO #to create an in-memory binary stream for storing the heatmap image.
import base64 # for encoding the heatmap image.

def generate_heatmap(request):
    # Retrieve data from the GazeData model
    gaze_data = GazeData.objects.all()

    # Convert queryset to a DataFrame
    df = pd.DataFrame(list(gaze_data.values()))

    # If timestamp is not already in datetime format, convert it
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create a heatmap for gaze_x
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['gaze_x']].transpose(), cmap='viridis', annot=True, fmt=".0f", ax=ax)
    ax.set_title('Gaze X Heatmap')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Gaze X')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode the image as base64
    heatmap_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Render the template with the heatmap image
    return render(request, 'heatmap_template.html', {'heatmap_image': heatmap_image})

@csrf_exempt
def receive_gaze_data(request):
    if request.method == 'POST':
        try:
            data = request.json()
            gaze_x = float(data.get('gaze_x'))
            gaze_y = float(data.get('gaze_y'))
            head_x = float(data.get('head_x'))
            head_y = float(data.get('head_y'))

            # Trigger the Celery task to process and save the gaze data
            process_gaze_data.delay(gaze_x, gaze_y, head_x, head_y)

            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return JsonResponse({'status': 'error', 'message': 'Unsupported method'})
    


def view_gaze_data(request):
    gaze_data = GazeData.objects.all()
    return render(request, 'view_gaze_data.html', {'gaze_data': gaze_data})



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import EyeTrackingData  # Assuming you have a model for storing the data
from django.core.files.base import ContentFile
import base64
@csrf_exempt
@require_POST
def save_video(request):
    try:
        # Get the video data from the request
        video_data = request.POST.get('videoData')

        # Decode the base64-encoded video data
        video_content = base64.b64decode(video_data)
        print('Video Content:', video_content)

        # Create a ContentFile from the decoded data
        video_file = ContentFile(video_content)

        # Save the video file to the database using your model
        EyeTrackingData.objects.create(video_file=video_file)

        return JsonResponse({'success': True, 'message': 'Video saved successfully'})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})






from django.shortcuts import render
import cv2
from keras.models import model_from_json
import numpy as np


json_file = open("evaluator/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("evaluator/emotiondetector.h5")

# Load the face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels for emotions
labels = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0




import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
from keras.models import model_from_json
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .models import EmotionData

# Your existing code for emotion detection and webcam setup

# Global variable to store emotion data
emotion_data = {'angry': 0, 'happy': 0, 'sad': 0, 'surprise': 0}

def detect_emotion(frame):
    global emotion_data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Update emotion data
        emotion_data[prediction_label] += 1

        cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {prediction_label}', (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    return frame

@gzip.gzip_page
def face_recognition(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    webcam = cv2.VideoCapture(0)

    while True:
        success, frame = webcam.read()

        if not success:
            break

        # Perform live emotion detection
        frame = detect_emotion(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    webcam.release()

# def generate_bar_chart(request):
#     global emotion_data

#     # Create a bar chart
#     emotions = list(emotion_data.keys())
#     counts = list(emotion_data.values())

#     plt.bar(emotions, counts)
#     plt.xlabel('Emotions')
#     plt.ylabel('Count')
#     plt.title('Emotion Detection Results')
    
#     # Save the chart to an image
#     image_stream = BytesIO()
#     plt.savefig(image_stream, format='png')
#     plt.close()

#     # Save the image to the database
#     image_data = image_stream.getvalue()
    
#     forwarded_data_instances = ForwardedData.objects.all()

#     for forwarded_data_instance in forwarded_data_instances:
#         emotion_data_instance = EmotionData(
#             website_name=forwarded_data_instance.website_name,
#             website_url=forwarded_data_instance.website_url,
#             description=forwarded_data_instance.description,
#             chart_image=image_data
#         )
#         emotion_data_instance.save()

    

#     return JsonResponse({'status': 'success'})



def show_chart(request):
    emotion_data = EmotionData.objects.all()  # Assuming you want to display the latest stored chart
    
    return render(request, 'show_chart.html', {'emotion_data': emotion_data})

def face(request, forwarded_data_id):
    # Retrieve the ForwardedData instance based on the provided ID
    forwarded_data = get_object_or_404(ForwardedData, id=forwarded_data_id)

    # Pass the website URL to the template
    context = {
        'forwarded_data_id': forwarded_data_id,
        'website_url': forwarded_data.website_url,
    }

    return render(request, 'face.html', context)

# def show_chart(request, project_id):
#     emotion_chart = get_object_or_404(EmotionData, project_id=project_id) # Assuming you want to display the latest stored chart
#     context = {'emotion_chart': emotion_chart}
#     return render(request, 'show_chart.html', context)

# Assuming you've imported necessary modules
from django.http import JsonResponse
from io import BytesIO
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import JsonResponse

# Assuming you've imported other necessary modules

def generate_bar_chart(request):
    global emotion_data

    # Set the backend to Agg
    plt.switch_backend('Agg')

    # Retrieve the ForwardedData instance based on the provided ID
    forwarded_data_id = request.GET.get('forwarded_data_id')

    # Retrieve the ForwardedData instance based on the provided ID
    forwarded_data_instance = get_object_or_404(ForwardedData, id=forwarded_data_id)

    # Create a bar chart for the specified ForwardedData instance
    emotions = list(emotion_data.keys())
    counts = list(emotion_data.values())

    plt.bar(emotions, counts)
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotion Detection Results')

    # Save the chart to an image
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Save the image to the database for the specified ForwardedData instance
    image_data = image_stream.getvalue()

    emotion_data_instance = EmotionData(
        website_name=forwarded_data_instance.website_name,
        website_url=forwarded_data_instance.website_url,
        description=forwarded_data_instance.description,
        chart_image=image_data
    )
    emotion_data_instance.save()

    return JsonResponse({'status': 'success'})


