import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from predictor import predictor  # Assuming predictor function is in predictor.py file

# Set window size
Window.size = (800, 600)

class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super(CaptureScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')

        # Create a button for uploading an image
        self.upload_button = Button(text='Upload Image', size_hint=(1, 0.1))
        self.upload_button.bind(on_press=self.upload_image)
        self.layout.add_widget(self.upload_button)

        self.add_widget(self.layout)

    def upload_image(self, instance):
        # Open the file chooser when the Upload Image button is pressed
        file_chooser = FileChooserIconView(path='.')
        file_chooser.bind(on_submit=self.load_uploaded_image)
        popup = Popup(title='Select an Image', content=file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_uploaded_image(self, file_chooser, selected_file, _):
        # Check if the selected file is a JPEG or PNG image
        if selected_file and (os.path.splitext(selected_file[0])[1].lower() == '.jpeg' or \
            os.path.splitext(selected_file[0])[1].lower() == '.png' or \
            os.path.splitext(selected_file[0])[1].lower() == '.jpg'):
            class_name, probability = predictor(selected_file[0], "class_dict.csv", "model.tflite")
            App.get_running_app().show_result(selected_file[0], class_name, probability)
        else:
            print("Please select a JPEG, JPG or PNG image.")



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
        # Change to the capture screen
        self.manager.current = 'capture_screen'

class SkinDiseaseDetectorApp(App):
    def __init__(self, **kwargs):
        super(SkinDiseaseDetectorApp, self).__init__(**kwargs)
        self.screen_manager = ScreenManager()

    def build(self):
        self.capture_screen = CaptureScreen(name='capture_screen')
        self.screen_manager.add_widget(self.capture_screen)
        return self.screen_manager

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
