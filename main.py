import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.clock import Clock
import cv2
from predictor import predictor  # Assuming predictor function is in predictor.py file

# Set window size
Window.size = (800, 600)

class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super(CaptureScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')

        # Create an image widget for displaying the camera feed
        self.camera_feed = KivyImage()
        self.layout.add_widget(self.camera_feed)

        # Create a button for capturing the image
        self.capture_button = Button(text='Capture Image', size_hint=(1, 0.1))
        self.capture_button.bind(on_press=self.capture_image)
        self.layout.add_widget(self.capture_button)

        # Create a button for uploading an image
        self.upload_button = Button(text='Upload Image', size_hint=(1, 0.1))
        self.upload_button.bind(on_press=self.upload_image)
        self.layout.add_widget(self.upload_button)

        # Create a file chooser for uploading image
        self.file_chooser = FileChooserListView(path='.')
        self.file_chooser.bind(on_submit=self.load_uploaded_image)
        self.layout.add_widget(self.file_chooser)

        self.add_widget(self.layout)

        # Schedule updating the camera feed
        Clock.schedule_interval(self.update_camera_feed, 1.0 / 30.0)

    def capture_image(self, instance):
        # Call the capture_image method of the App
        App.get_running_app().capture_image()

    def upload_image(self, instance):
        # Open the file chooser when the Upload Image button is pressed
        self.file_chooser.path = '.'
        self.file_chooser.selection = []

    def load_uploaded_image(self, file_chooser, selected_file, haha):
        # Check if the selected file is a JPEG or PNG image
        if selected_file and (os.path.splitext(selected_file[0])[1].lower() == '.jpeg' or \
            os.path.splitext(selected_file[0])[1].lower() == '.png' or \
            os.path.splitext(selected_file[0])[1].lower() == '.jpg'):
            class_name, probability = predictor(selected_file[0], "class_dict.csv", "model.h5")
            App.get_running_app().show_result(selected_file[0], class_name, probability)
            self.camera_feed.source = selected_file[0]
        else:
            print("Please select a JPEG, JPG or PNG image.")

    def update_camera_feed(self, dt):
        # Read a frame from the camera
        ret, frame = App.get_running_app().capture.read()
        if ret:
            # Convert the frame to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Update the image widget with the new texture
            self.camera_feed.texture = texture


    def update_camera_feed(self, dt):
        # Read a frame from the camera
        ret, frame = App.get_running_app().capture.read()
        if ret:
            # Convert the frame to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Update the image widget with the new texture
            self.camera_feed.texture = texture

class ResultScreen(Screen):
    def __init__(self, image_path, class_name, probability, **kwargs):
        super(ResultScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')

        # Display the captured image
        image = KivyImage(source=image_path, size_hint=(1, 0.8))
        self.layout.add_widget(image)

        # Display the predicted class and probability
        prediction_label = Label(text=f'Skin Disease is {class_name} with a probability of {probability * 100: 6.2f} %', size_hint=(1, 0.1))
        self.layout.add_widget(prediction_label)

        # Create a button to go back to the capture screen
        back_button = Button(text='Capture New Image', size_hint=(1, 0.1))
        back_button.bind(on_press=self.go_to_capture_screen)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def go_to_capture_screen(self, instance):
        # Release the existing camera capture object
        App.get_running_app().capture.release()
        # Recreate the camera capture object
        App.get_running_app().capture = cv2.VideoCapture(0)
        # Change to the capture screen
        self.manager.current = 'capture_screen'

class SkinDiseaseDetectorApp(App):
    def __init__(self, **kwargs):
        super(SkinDiseaseDetectorApp, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)  # Camera capture
        self.screen_manager = ScreenManager()

    def build(self):
        self.capture_screen = CaptureScreen(name='capture_screen')
        self.screen_manager.add_widget(self.capture_screen)
        return self.screen_manager

    def capture_image(self):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("captured_image.jpg", frame)  # Save captured image
            class_name, probability = predictor("captured_image.jpg", "class_dict.csv", "model.h5")
            self.show_result("captured_image.jpg", class_name, probability)

    def show_result(self, image_path, class_name, probability):
        # Remove the old result screen if it exists
        if 'result_screen' in self.screen_manager.screen_names:
            self.screen_manager.remove_widget(self.screen_manager.get_screen('result_screen'))

        # Add the new result screen
        result_screen = ResultScreen(image_path=image_path, class_name=class_name, probability=probability, name='result_screen')
        self.screen_manager.add_widget(result_screen)
        self.screen_manager.current = 'result_screen'


if __name__ == '__main__':
    SkinDiseaseDetectorApp().run()
