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
from kivy.graphics import Rectangle, Color
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from jinus import TensorFlowModel
from PIL import Image
import numpy as np

class StartScreen(Screen):
    def __init__(self, **kwargs):
        super(StartScreen, self).__init__(**kwargs)
        self.background_color = [1, 1, 1, 1]
        self.layout = BoxLayout(orientation='vertical')

        # Add image covering the whole screen
        image = KivyImage(source='first_page.jpg', size=(1, 1))
        self.layout.add_widget(image)

        # Add next button
        next_button = Button(
            text='Next',
            size_hint=(None, None),
            size=(800, 200),
            pos_hint={'center_x': 0.5},
            background_color=(1, 1, 1, 1),
            color=(0, 0, 0, 1)
        )
        next_button.bind(on_press=self.go_to_capture_screen)

        self.layout.add_widget(next_button)

        self.add_widget(self.layout)


    def go_to_capture_screen(self, instance):
        self.manager.current = 'capture_screen'

class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super(CaptureScreen, self).__init__(**kwargs)
        # Create a white background widget
        background = Widget()
        background.canvas.add(Color(1, 1, 1, 1))
        background.canvas.add(Rectangle(size=self.size))
        
        # Create a BoxLayout with vertical orientation to contain the button
        self.layout = BoxLayout(orientation='vertical', padding=50, spacing=50, size_hint_y=None)
        
        # Add the upload button
        self.upload_button = Button(
            text='Upload Image',
            size_hint=(None, None),
            size=(800, 200),
            pos_hint={'center_x': 0.5},
            background_color=(1, 1, 1, 1),  # Light blue background color
            color=(0, 0, 0, 1),  # Black text color
        )

        # Bind the button press event
        self.upload_button.bind(on_press=self.upload_image)

        # Add the button to the layout
        self.layout.add_widget(self.upload_button)

        # Set the BoxLayout's height to match its content
        self.layout.bind(minimum_height=self.layout.setter('height'))

        # Position the BoxLayout in the center of the screen vertically
        self.layout.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        # Add the layout to the screen
        self.add_widget(self.layout)

    def upload_image(self, instance):
        # Open the file chooser when the Upload Image button is pressed
        file_chooser = FileChooserIconView(path='.')
        file_chooser.bind(on_submit=self.load_uploaded_image)
        popup = Popup(title='Select an Image', content=file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_uploaded_image(self, file_chooser, selected_file, _):
        # Check if the selected file is a JPEG or PNG image
        print(selected_file)
        if selected_file and (os.path.splitext(selected_file[0])[1].lower() == '.jpeg' or \
            os.path.splitext(selected_file[0])[1].lower() == '.png' or \
            os.path.splitext(selected_file[0])[1].lower() == '.jpg'):
            butterflies = [
                'BA- cellulitis', 
                'BA-impetigo', 
                'FU-athlete-foot', 
                'FU-nail-fungus', 
                'FU-ringworm', 
                'PA-cutaneous-larva-migrans', 
                'VI-chickenpox', 
                'VI-shingles']
    
            # Load TFLite model
            model_to_pred = TensorFlowModel()
            model_to_pred.load(os.path.join(os.getcwd(), 'model.tflite'))
    
            # Read image and predict
            img = Image.open(selected_file[0])
            img_arr = np.array(img.resize((300, 300)), np.float32)
            img_arr = img_arr[:, :, :3]
            img_arr = np.expand_dims(img_arr, axis=0)
            preds = dict(zip(butterflies, list(model_to_pred.pred(img_arr)[0])))
            best = max(preds, key=preds.get)
            App.get_running_app().show_result(selected_file[0], best, str(preds[best]*100))
        else:
            print("Please select a JPEG, JPG or PNG image.")



class ResultScreen(Screen):
    def __init__(self, image_path, class_name, probability, **kwargs):
        super(ResultScreen, self).__init__(**kwargs)
        self.background_color = [1, 1, 1, 1]
        self.layout = BoxLayout(orientation='vertical')

        # Display the captured image
        image = KivyImage(source=image_path, size_hint=(1, 1))
        self.layout.add_widget(image)

        # Display the predicted class and probability
        prediction_label = result_label = Label(text=f'Disease: {class_name}\nProbability: {probability}', size_hint=(1, 0.1))  
        self.layout.add_widget(prediction_label)

        # Create a button to go back to the capture screen
        back_button = Button(
            text='Capture New Image',
            size_hint=(None, None),
            size=(800, 200),
            pos_hint={'center_x': 0.5},
            background_color=(1, 1, 1, 1),  # Light blue background color
            color=(0, 0, 0, 1),  # Black text color
        )
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
        self.start_screen = StartScreen(name='start_screen')
        self.capture_screen = CaptureScreen(name='capture_screen')
        self.screen_manager.add_widget(self.start_screen)
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
