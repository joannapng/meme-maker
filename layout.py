import ipywidgets as widgets
from ipywidgets import interactive
import config as cfg
from IPython.display import display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import imageio.v2
from filters import *
from face_detection import *
from eyes_detection import *
from meme_maker import *
from ipywidgets import GridspecLayout

class MemeAssetLayout:
    """
    Class that adds meme assets
    """
    def __init__(self, uploader, filters):
        self.uploader = uploader # Upload file tab
        self.filters = filters # Filters tab
        self.img = None
        self.tmp_img = None # img before committing changes
        self.face_choice = -1 # which face to add asset to
        self.asset_choice = -1 # which asset to add
        self.asset_scale = 1 
        self.offset_y = 0 # vertical asset offset
        self.offset_x = 0 # horizontal asset offset
        self.flip_x = 0 # horizontal flip
        self.flip_y = 0 # vertical flip

        # ------ Buttons and Sliders ------ # 
        self.done_btn = widgets.Button(
            value = False, 
            description = 'Done', 
            disabled = False, 
            button_style = 'Success',
            icon = 'check'
        )

        self.asset_scale_slider = widgets.FloatSlider(
            value = 1, 
            min = 0.1,
            max = 10,
            step = 0.1,
            description = 'Asset Scale',
            disabled = False,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = '.1f'
        )

        self.asset_horizontal_slider = widgets.IntSlider(
            value=0,
            min=-1000,
            max=1000,
            step=1,
            description='Horizontal shift',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.asset_vertical_slider = widgets.IntSlider(
            value = 0, 
            min = -1000, 
            max = 1000, 
            step = 1, 
            description = 'Vertical shift',
            disabled = False, 
            continuous_update = False,
            orientation = 'horizontal',
            readout = False
        )

        self.asset_hflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset horizontally',
            disabled = False,
            indent = False
        )

        self.asset_vflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset vertically',
            disabled = False,
            indent = False
        )

        # keep buttons and sliders hidden before asset is selected
        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_scale_slider.observe(self.asset_scale_handler, names = 'value')
        
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.observe(self.asset_horizontal_slider_handler, names = 'value')

        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.observe(self.asset_vertical_slider_handler, names = 'value')

        self.asset_hflip_checker.layout.visibility = 'hidden'
        self.asset_hflip_checker.observe(self.asset_hflip_checker_handler, names = 'value')
        
        self.asset_vflip_checker.layout.visibility = 'hidden'
        self.asset_vflip_checker.observe(self.asset_vflip_checker_handler, names = 'value')

        self.done_btn.layout.visibility = 'hidden'
        self.done_btn.on_click(self.done_btn_handler)

        # edit canvas
        self.detection_output = widgets.Output()
        self.outputs = [self.detection_output]

        # wait for a new img to be uploaded
        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')        
        self.btns = self.filters.done_btns

        # choose face
        self.detections = ()
        self.choose_face_dropdown = widgets.Dropdown(options = [])
        self.choose_face_accordion = widgets.Accordion(children=[self.choose_face_dropdown], titles=['Choose face to add meme asset'])
        self.choose_face_dropdown.observe(self.choose_face_dropdown_handler, names = 'value')
        
        self.done_btns = [self.done_btn]
        
        # 2 row, 3 column asset display (print available assets)
        self.assets = GridspecLayout(2, 3)

        for i, asset in enumerate(cfg.meme_face_assets):
            self.assets[i//3, i%3] = widgets.VBox(children=[widgets.Label(value=str(i+1)), widgets.Output()])
            with self.assets[i//3, i%3].children[-1]:
                loaded_img = imageio.imread(asset)
                img_plot = plt.imshow(loaded_img)
                img_plot = plt.axis('off')
                plt.show()

        # update img if filter done btn is pressed
        for btn in self.btns:
            btn.on_click(self.done_filter_btn_handler)

        # choose asset
        self.choose_asset_dropdown = widgets.Dropdown(options = [(str(i+1), i) for i in range(len(cfg.meme_face_assets))], value = None, disabled=True)
        self.choose_asset_dropdown.observe(self.choose_asset_dropdown_handler, names = 'value')
        self.choose_asset_layout = widgets.VBox(children=[self.choose_asset_dropdown, self.assets])
        self.choose_asset_accordion = widgets.Accordion(children=[self.choose_asset_layout], titles = ['Choose meme asset'])
        self.selections = widgets.VBox(children=[self.choose_face_accordion, self.choose_asset_accordion])
        
        # layout
        self.edit_canvas = widgets.Output()
        self.edit_tools_layout = widgets.VBox(children=[self.asset_scale_slider, self.asset_horizontal_slider, self.asset_vertical_slider, self.asset_hflip_checker, self.asset_vflip_checker, self.done_btn])
        self.edit_canvas_layout = widgets.HBox(children=[self.edit_canvas, self.edit_tools_layout])
        self.layout = widgets.VBox(children=[self.detection_output, self.selections, self.edit_canvas_layout])
   
    def new_img_output_handler(self, change):
        """
        Update canvas when a new image is uploaded
        """
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']

        if self.uploader.uploaded_image is not None  and self.uploaded_file_type in cfg.supported_types_list:
            self.img = self.uploader.uploaded_image
        else:
            self.img = None
        
        for output in self.outputs:
            with output:
                if self.img is not None:
                    output.clear_output(wait=True)
                    # perform face detection and draw bounding boxes
                    self.img_w_boxes, self.detections = face_detection(self.img)
                    self.num_detections = len(self.detections)
                    self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
                    img_plot = plt.imshow(self.img_w_boxes)
                    img_plot = plt.axis('off')
                    plt.show()    
                else:
                    output.clear_output(wait=False)        
        
        # reset controls to initial state
        self.reset()

    # when done button from filters tab is pressed, update img
    def done_filter_btn_handler(self, btn):
        """
        Update canvas when a done filter btn is pressed
        """
        self.img = self.filters.tmp_img
        with self.detection_output:
            self.detection_output.clear_output(wait=True)
            # perform face detection and draw bounding boxes
            self.img_w_boxes, self.detections = face_detection(self.img)
            self.num_detections = len(self.detections)
            self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
            img_plot = plt.imshow(self.img_w_boxes)
            img_plot = plt.axis('off')
            plt.show()

        self.reset()

    def choose_face_dropdown_handler(self, change):
        """
        Choose face to add meme asset to from dropdown menu
        """
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            if change['new'] is not None:
                self.face_choice = change['new']
                self.choose_asset_dropdown.disabled = False
                img_plot = plt.imshow(self.img)
                img_plot = plt.axis('off')
                plt.show()

    def choose_asset_dropdown_handler(self, change):
        """
        Choose asset from dropdown menu
        """
        if self.img is not None:
            self.asset_scale_slider.layout.visibility = 'visible'
            self.asset_horizontal_slider.layout.visibility = 'visible'
            self.asset_vertical_slider.layout.visibility = 'visible'
            self.asset_hflip_checker.layout.visibility = 'visible'
            self.asset_vflip_checker.layout.visibility = 'visible'
            self.done_btn.layout.visibility = 'visible'
        else:
            self.asset_scale_slider.layout.visibility = 'hidden'
            self.asset_horizontal_slider.layout.visibility = 'hidden'
            self.asset_vertical_slider.layout.visibility = 'hidden'
            self.asset_hflip_checker.layout.visibility = 'hidden'
            self.asset_vflip_checker.layout.visibility = 'hidden'
            self.done_btn.layout.visibility = 'hidden'

        H, W, _ = self.img.shape
        self.asset_horizontal_slider.min = -H //2 
        self.asset_horizontal_slider.max = H // 2
        self.asset_vertical_slider.min = -W // 2
        self.asset_vertical_slider.max = W // 2

        with self.edit_canvas:
            # add asset only if a face has been chosen
            if change['new'] is not None and self.face_choice != -1 and self.img is not None:
                self.asset_choice = change['new']
                self.edit_canvas.clear_output(wait = True)
                self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice, 
                                            self.asset_choice, self.img)
                img_plot = plt.imshow(self.tmp_img)
                img_plot = plt.axis('off')
                plt.show()
        
    def done_btn_handler(self, btn):
        """
        Update img when the done button is pressed
        """
        if self.tmp_img is not None:
            self.img = self.tmp_img

    def asset_scale_handler(self, change):
        """
        Handler invoked when an asset scale is chosen
        """
        self.asset_scale = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_horizontal_slider_handler(self, change):
        """
        Handler invoked when a value from the horizontal slider is chosen
        """
        self.offset_x= change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vertical_slider_handler(self, change):
        """
        Handler invoked when a value from the vertical slider is chosen
        """
        self.offset_y = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_hflip_checker_handler(self, change):
        """
        Handler invoked when hflip checker state changes
        """
        self.flip_x = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vflip_checker_handler(self, change):
        """
        Handler invoked when vflip checker state changes
        """
        self.flip_y = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_asset(cfg.meme_face_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def reset(self):
        """
        Reset when a new image is uploaded
        """
        self.edit_canvas.clear_output(wait=False)
        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.done_btn.layout.visibility = 'hidden'
        self.choose_face_dropdown.value = None
        self.choose_asset_dropdown.value = None
        self.face_choice = -1
        self.asset_choice = -1
        self.choose_asset_dropdown.disabled = True
        self.asset_scale_slider.value = 1
        self.asset_horizontal_slider.value = 0
        self.asset_vertical_slider.value = 0
        self.asset_hflip_checker.value = False
        self.asset_vflip_checker.value = False
        self.offset_x = 0
        self.offset_y = 0
        self.flip_x = 0
        self.flip_y = 0

class AddGlassesLayout:
    """
    Class for adding glasses, same functions as above, replace face
    detecction and eye detection
    """
    def __init__(self, uploader, filters):
        self.uploader = uploader
        self.filters = filters
        self.img = None
        self.tmp_img = None
        self.eyes_choice = -1
        self.asset_choice = -1
        self.asset_scale = 1
        self.offset_y = 0
        self.offset_x = 0
        self.flip_x = 0
        self.flip_y = 0

        self.done_btn = widgets.Button(
            value = False, 
            description = 'Done', 
            disabled = False, 
            button_style = 'Success',
            icon = 'check'
        )

        self.asset_scale_slider = widgets.FloatSlider(
            value = 1, 
            min = 0.1,
            max = 10,
            step = 0.1,
            description = 'Asset Scale',
            disabled = False,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = '.1f'
        )

        self.asset_horizontal_slider = widgets.IntSlider(
            value=0,
            min=-128,
            max=128,
            step=1,
            description='Horizontal shift',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.asset_vertical_slider = widgets.IntSlider(
            value = 0, 
            min = -128, 
            max = 128, 
            step = 1, 
            description = 'Vertical shift',
            disabled = False, 
            continuous_update = False,
            orientation = 'horizontal',
            readout = False
        )

        self.asset_hflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset horizontally',
            disabled = False,
            indent = False
        )

        self.asset_vflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset vertically',
            disabled = False,
            indent = False
        )

        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_scale_slider.observe(self.asset_scale_handler, names = 'value')
        
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.observe(self.asset_horizontal_slider_handler, names = 'value')

        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.observe(self.asset_vertical_slider_handler, names = 'value')

        self.asset_hflip_checker.layout.visibility = 'hidden'
        self.asset_hflip_checker.observe(self.asset_hflip_checker_handler, names = 'value')
        
        self.asset_vflip_checker.layout.visibility = 'hidden'
        self.asset_vflip_checker.observe(self.asset_vflip_checker_handler, names = 'value')

        self.done_btn.layout.visibility = 'hidden'
        self.done_btn.on_click(self.done_btn_handler)

        self.detection_output = widgets.Output()
        self.outputs = [self.detection_output]

        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')        
        self.btns = self.filters.done_btns

        self.detections = ()
        self.choose_face_dropdown = widgets.Dropdown(options = [])
        self.choose_face_accordion = widgets.Accordion(children=[self.choose_face_dropdown], titles=['Choose face to add meme asset'])
        self.choose_face_dropdown.observe(self.choose_face_dropdown_handler, names = 'value')
        self.assets = GridspecLayout(1, 2)

        self.done_btns = [self.done_btn]

        for i, asset in enumerate(cfg.eye_assets):
            self.assets[i//2, i%2] = widgets.VBox(children=[widgets.Label(value=str(i+1)), widgets.Output()])
            with self.assets[i//2, i%2].children[-1]:
                loaded_img = imageio.imread(asset)
                img_plot = plt.imshow(loaded_img)
                img_plot = plt.axis('off')
                plt.show()

        for btn in self.btns:
            btn.on_click(self.done_filter_btn_handler)

        self.choose_asset_dropdown = widgets.Dropdown(options = [(str(i+1), i) for i in range(len(cfg.eye_assets))], value = None, disabled=True)
        self.choose_asset_dropdown.observe(self.choose_asset_dropdown_handler, names = 'value')
        self.choose_asset_layout = widgets.VBox(children=[self.choose_asset_dropdown, self.assets])
        self.choose_asset_accordion = widgets.Accordion(children=[self.choose_asset_layout], titles = ['Choose meme asset'])
        self.selections = widgets.VBox(children=[self.choose_face_accordion, self.choose_asset_accordion])
        
        self.edit_canvas = widgets.Output()
        self.edit_tools_layout = widgets.VBox(children=[self.asset_scale_slider, self.asset_horizontal_slider, self.asset_vertical_slider, self.asset_hflip_checker, self.asset_vflip_checker, self.done_btn])
        self.edit_canvas_layout = widgets.HBox(children=[self.edit_canvas, self.edit_tools_layout])
        self.layout = widgets.VBox(children=[self.detection_output, self.selections, self.edit_canvas_layout])

    def new_img_output_handler(self, change):
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']

        if self.uploader.uploaded_image is not None  and self.uploaded_file_type in cfg.supported_types_list:
            self.img = self.uploader.uploaded_image
        else:
            self.img = None
        
        for output in self.outputs:
            with output:
                if self.img is not None:
                    output.clear_output(wait=True)
                    self.img_w_boxes, self.detections = eyes_detection(self.img)
                    self.num_detections = len(self.detections)
                    self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
                    img_plot = plt.imshow(self.img_w_boxes)
                    img_plot = plt.axis('off')
                    plt.show()        
                else:
                    output.clear_output(wait=False)    

        self.reset()

    def done_filter_btn_handler(self, btn):
        self.img = self.filters.tmp_img
        with self.detection_output:
            self.detection_output.clear_output(wait=True)
            self.img_w_boxes, self.detections = eyes_detection(self.img)
            self.num_detections = len(self.detections)
            self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
            img_plot = plt.imshow(self.img_w_boxes)
            img_plot = plt.axis('off')
            plt.show()

        self.reset()

    def choose_face_dropdown_handler(self, change):
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            if change['new'] is not None:
                self.eyes_choice = change['new']
                self.choose_asset_dropdown.disabled = False
                img_plot = plt.imshow(self.img)
                img_plot = plt.axis('off')
                plt.show()

    def choose_asset_dropdown_handler(self, change):
        if self.img is not None:
            self.asset_scale_slider.layout.visibility = 'visible'
            self.asset_horizontal_slider.layout.visibility = 'visible'
            self.asset_vertical_slider.layout.visibility = 'visible'
            self.asset_hflip_checker.layout.visibility = 'visible'
            self.asset_vflip_checker.layout.visibility = 'visible'
            self.done_btn.layout.visibility = 'visible'
        else:
            self.asset_scale_slider.layout.visibility = 'hidden'
            self.asset_horizontal_slider.layout.visibility = 'hidden'
            self.asset_vertical_slider.layout.visibility = 'hidden'
            self.asset_hflip_checker.layout.visibility = 'hidden'
            self.asset_vflip_checker.layout.visibility = 'hidden'
            self.done_btn.layout.visibility = 'hidden'

        H, W, _ = self.img.shape
        self.asset_horizontal_slider.min = -H // 2
        self.asset_horizontal_slider.max = H // 2
        self.asset_vertical_slider.min = -W // 2
        self.asset_vertical_slider.max = W // 2
        
        with self.edit_canvas:
            if change['new'] is not None and self.eyes_choice != -1 and self.img is not None:
                self.asset_choice = change['new']
                self.edit_canvas.clear_output(wait = True)
                self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.eyes_choice, 
                                            self.asset_choice, self.img)
                img_plot = plt.imshow(self.tmp_img)
                img_plot = plt.axis('off')
                plt.show()
        
    def done_btn_handler(self, btn):
        if self.tmp_img is not None:
            self.img = self.tmp_img

    def asset_scale_handler(self, change):
        self.asset_scale = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_horizontal_slider_handler(self, change):
        self.offset_x= change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vertical_slider_handler(self, change):
        self.offset_y= change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_hflip_checker_handler(self, change):
        self.flip_x = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vflip_checker_handler(self, change):
        self.flip_y = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_asset(cfg.eye_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, bounding_box_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()
    
    def reset(self):
        self.edit_canvas.clear_output(wait=False)
        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.done_btn.layout.visibility = 'hidden'
        self.choose_face_dropdown.value = None
        self.choose_asset_dropdown.value = None
        self.face_choice = -1
        self.asset_choice = -1
        self.choose_asset_dropdown.disabled = True
        self.asset_scale_slider.value = 1
        self.asset_horizontal_slider.value = 0
        self.asset_vertical_slider.value = 0
        self.asset_hflip_checker.value = False
        self.asset_vflip_checker.value = False
        self.offset_x = 0
        self.offset_y = 0
        self.flip_x = 0
        self.flip_y = 0
            
class AddHatLayout:
    def __init__(self, uploader, filters):
        self.uploader = uploader = uploader 
        self.filters = filters
        self.img = self.uploader.uploaded_image
        self.tmp_img = None
        self.eyes_choice = -1
        self.asset_choice = -1
        self.asset_scale = 1
        self.offset_y = 0
        self.offset_x = 0
        self.flip_x = 0
        self.flip_y = 0

        self.done_btn = widgets.Button(
            value = False, 
            description = 'Done', 
            disabled = False, 
            button_style = 'Success',
            icon = 'check'
        )

        self.asset_scale_slider = widgets.FloatSlider(
            value = 1, 
            min = 0.1,
            max = 10,
            step = 0.1,
            description = 'Asset Scale',
            disabled = False,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = '.1f'
        )

        self.asset_horizontal_slider = widgets.IntSlider(
            value=0,
            min=-128,
            max=128,
            step=1,
            description='Horizontal shift',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.asset_vertical_slider = widgets.IntSlider(
            value = 0, 
            min = -128, 
            max = 128, 
            step = 1, 
            description = 'Vertical shift',
            disabled = False, 
            continuous_update = False,
            orientation = 'horizontal',
            readout = False
        )

        self.asset_hflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset horizontally',
            disabled = False,
            indent = False
        )

        self.asset_vflip_checker = widgets.Checkbox(
            value = False,
            description = 'Flip asset vertically',
            disabled = False,
            indent = False
        )

        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_scale_slider.observe(self.asset_scale_handler, names = 'value')
        
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.observe(self.asset_horizontal_slider_handler, names = 'value')

        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.observe(self.asset_vertical_slider_handler, names = 'value')

        self.asset_hflip_checker.layout.visibility = 'hidden'
        self.asset_hflip_checker.observe(self.asset_hflip_checker_handler, names = 'value')
        
        self.asset_vflip_checker.layout.visibility = 'hidden'
        self.asset_vflip_checker.observe(self.asset_vflip_checker_handler, names = 'value')

        self.done_btn.layout.visibility = 'hidden'
        self.done_btn.on_click(self.done_btn_handler)

        self.detection_output = widgets.Output()
        self.outputs = [self.detection_output]

        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')        
        self.btns = self.filters.done_btns

        self.detections = ()
        self.choose_face_dropdown = widgets.Dropdown(options = [])
        self.choose_face_accordion = widgets.Accordion(children=[self.choose_face_dropdown], titles=['Choose face to add meme asset'])
        self.choose_face_dropdown.observe(self.choose_face_dropdown_handler, names = 'value')
        self.assets = GridspecLayout(1, 2)

        self.done_btns = [self.done_btn]

        for i, asset in enumerate(cfg.hat_assets):
            self.assets[i//2, i%2] = widgets.VBox(children=[widgets.Label(value=str(i+1)), widgets.Output()])
            with self.assets[i//2, i%2].children[-1]:
                loaded_img = imageio.imread(asset)
                img_plot = plt.imshow(loaded_img)
                img_plot = plt.axis('off')
                plt.show()

        for btn in self.btns:
            btn.on_click(self.done_filter_btn_handler)

        self.choose_asset_dropdown = widgets.Dropdown(options = [(str(i+1), i) for i in range(len(cfg.eye_assets))], value = None)
        self.choose_asset_dropdown.observe(self.choose_asset_dropdown_handler, names = 'value')
        self.choose_asset_layout = widgets.VBox(children=[self.choose_asset_dropdown, self.assets])
        self.choose_asset_accordion = widgets.Accordion(children=[self.choose_asset_layout], titles = ['Choose meme asset'])
        self.selections = widgets.VBox(children=[self.choose_face_accordion, self.choose_asset_accordion])
        
        self.edit_canvas = widgets.Output()
        self.edit_tools_layout = widgets.VBox(children=[self.asset_scale_slider, self.asset_horizontal_slider, self.asset_vertical_slider, self.asset_hflip_checker, self.asset_vflip_checker, self.done_btn])
        self.edit_canvas_layout = widgets.HBox(children=[self.edit_canvas, self.edit_tools_layout])
        self.layout = widgets.VBox(children=[self.detection_output, self.selections, self.edit_canvas_layout])

    def new_img_output_handler(self, change):
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']

        if self.uploader.uploaded_image is not None  and self.uploaded_file_type in cfg.supported_types_list:
            self.img = self.uploader.uploaded_image
        else:
            self.img = None
        
        for output in self.outputs:
            with output:
                if self.img is not None:
                    output.clear_output(wait=True)
                    self.img_w_boxes, self.detections = face_detection(self.img)
                    self.num_detections = len(self.detections)
                    self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
                    img_plot = plt.imshow(self.img_w_boxes)
                    img_plot = plt.axis('off')
                    plt.show()           
                else:
                    output.clear_output(wait=False) 

        self.reset()

    def done_filter_btn_handler(self, btn):
        self.img = self.filters.tmp_img
        with self.detection_output:
            self.detection_output.clear_output(wait=True)
            self.img_w_boxes, self.detections = face_detection(self.img)
            self.num_detections = len(self.detections)
            self.choose_face_dropdown.options = [(str(i+1), i) for i in range(self.num_detections)]
            img_plot = plt.imshow(self.img_w_boxes)
            img_plot = plt.axis('off')
            plt.show()

        self.reset()

    def choose_face_dropdown_handler(self, change):
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            if change['new'] is not None:
                self.eyes_choice = change['new']
                self.choose_asset_dropdown.disabled = False
                img_plot = plt.imshow(self.img)
                img_plot = plt.axis('off')
                plt.show()

    def choose_asset_dropdown_handler(self, change):
        if self.img is not None:
            self.asset_scale_slider.layout.visibility = 'visible'
            self.asset_horizontal_slider.layout.visibility = 'visible'
            self.asset_vertical_slider.layout.visibility = 'visible'
            self.asset_hflip_checker.layout.visibility = 'visible'
            self.asset_vflip_checker.layout.visibility = 'visible'
            self.done_btn.layout.visibility = 'visible'
        else:
            self.asset_scale_slider.layout.visibility = 'hidden'
            self.asset_horizontal_slider.layout.visibility = 'hidden'
            self.asset_vertical_slider.layout.visibility = 'hidden'
            self.asset_hflip_checker.layout.visibility = 'hidden'
            self.asset_vflip_checker.layout.visibility = 'hidden'
            self.done_btn.layout.visibility = 'hidden'

        H, W, _ = self.img.shape
        self.asset_horizontal_slider.min = -H // 2
        self.asset_horizontal_slider.max = H // 2
        self.asset_vertical_slider.min = -W // 2
        self.asset_vertical_slider.max = W // 2
        
        with self.edit_canvas:
            if change['new'] is not None and self.eyes_choice != -1 and self.img is not None:
                self.asset_choice = change['new']
                self.edit_canvas.clear_output(wait = True)
                self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.eyes_choice, 
                                            self.asset_choice, self.img)
                img_plot = plt.imshow(self.tmp_img)
                img_plot = plt.axis('off')
                plt.show()
        
    def done_btn_handler(self, btn):
        if self.tmp_img is not None:
            self.img = self.tmp_img

    def asset_scale_handler(self, change):
        self.asset_scale = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, asset_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_horizontal_slider_handler(self, change):
        self.offset_x= change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, asset_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vertical_slider_handler(self, change):
        self.offset_y= change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait=True)
            self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.eyes_choice, 
                                     self.asset_choice, self.img, offset_y = self.offset_y, 
                                     offset_x = self.offset_x, asset_scale = self.asset_scale,
                                     flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()
    
    def asset_hflip_checker_handler(self, change):
        self.flip_x = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, asset_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def asset_vflip_checker_handler(self, change):
        self.flip_y = change['new']
        with self.edit_canvas:
            self.edit_canvas.clear_output(wait = True)
            self.tmp_img = add_hat(cfg.hat_assets, self.detections, self.face_choice,
                                    self.asset_choice, self.img, offset_y = self.offset_y, 
                                    offset_x = self.offset_x, asset_scale = self.asset_scale,
                                    flip_x = self.flip_x, flip_y = self.flip_y)
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def reset(self):
        self.edit_canvas.clear_output(wait=False)
        self.asset_scale_slider.layout.visibility = 'hidden'
        self.asset_horizontal_slider.layout.visibility = 'hidden'
        self.asset_vertical_slider.layout.visibility = 'hidden'
        self.done_btn.layout.visibility = 'hidden'
        self.choose_face_dropdown.value = None
        self.choose_asset_dropdown.value = None
        self.face_choice = -1
        self.asset_choice = -1
        self.choose_asset_dropdown.disabled = True
        self.asset_scale_slider.value = 1
        self.asset_horizontal_slider.value = 0
        self.asset_vertical_slider.value = 0
        self.asset_hflip_checker.value = False
        self.asset_vflip_checker.value = False
        self.offset_x = 0
        self.offset_y = 0
        self.flip_x = 0
        self.flip_y = 0

class MemeMakerLayout:
    def __init__(self, uploader, filters):
        """
        Class that instatiates the 3 classes (for meme assets, eye assets, hat assets)
        """
        self.uploader = uploader
        self.filters = filters
        self.img = self.uploader.uploaded_image
        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')
        self.meme_assets_layout = MemeAssetLayout(self.uploader, self.filters)
        self.add_glasses_layout = AddGlassesLayout(self.uploader, self.filters)
        self.add_hat_layout = AddHatLayout(self.uploader, self.filters)
        self.done_btns = []

        for btn in self.filters.done_btns:
            btn.on_click(self.done_filter_btn_handler)
            self.done_btns.append(btn)

        for btn in self.meme_assets_layout.done_btns:
            btn.on_click(self.done_meme_asset_btn_handler)
            self.done_btns.append(btn)

        for btn in self.add_glasses_layout.done_btns:
            btn.on_click(self.done_glasses_btn_handler)
            self.done_btns.append(btn)

        for btn in self.add_hat_layout.done_btns:
            btn.on_click(self.done_hat_btn_handler)
            self.done_btns.append(btn)
        
        self.layout = widgets.Accordion(children=[self.meme_assets_layout.layout, self.add_glasses_layout.layout, self.add_hat_layout.layout], 
                                        titles=['Add Meme Asset', 'Add Glasses', 'Add Hat'])

    def new_img_output_handler(self, change):
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']
        self.img = self.uploader.uploaded_image

    def done_filter_btn_handler(self, btn):
        self.img = self.filters.tmp_img

    def done_meme_asset_btn_handler(self, btn):
        self.img = self.meme_assets_layout.tmp_img
    
    def done_glasses_btn_handler(self, btn):
        self.img = self.add_glasses_layout.tmp_img

    def done_hat_btn_handler(self, btn):
        self.img = self.add_hat_layout.tmp_img

class SaveImageLayout:
    """
    Class that realizes the Save Image As layout
    """
    def __init__(self, uploader, editor):
        self.uploader = uploader
        self.editor = editor # editor is the last tab that has edited the picture
        self.img = self.editor.img

        # Preview
        self.preview = widgets.Output()

        # Text 
        self.filename = widgets.Text(
            value = '',
            placeholder = '',
            description = 'Save as: ',
            disabled = False
        )

        # save button
        self.save_btn = widgets.Button(
            value = False, 
            description = 'Save', 
            disabled = False, 
            button_style = '',
            icon = 'check'
        )
        self.save_btn.on_click(self.save_btn_handler)

        self.btns = self.editor.done_btns

        for btn in self.btns:
            btn.on_click(self.done_btn_handler)

        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')        
        self.layout = widgets.VBox(children=[self.preview, self.filename, self.save_btn])

    def new_img_output_handler(self, change):
        """
        Handler for the uploaded
        """
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']
        
        if self.uploader.uploaded_image is not None and self.uploaded_file_type in cfg.supported_types_list:
            # display controls if the image type is supported
            self.img = self.new_img = self.uploader.uploaded_image
        else:
            # if wrong type is uploaded hide the buttons and the controls
            self.img = self.new_img = None

    def done_btn_handler(self, btn):
        self.img = self.editor.img
        with self.preview:
            self.preview.clear_output(wait=True)
            img_plot = plt.imshow(self.img)
            img_plot = plt.axis('off')
            plt.show()

    def save_btn_handler(self, btn):
        self.path = self.filename.value
        self.save_img()

    def save_img(self):
        if self.img is not None:
            image = Image.fromarray(self.img)
            image.save(f'Images/{self.filename.value}.jpeg', format="JPEG")


class FilterLayout:
    """
    Class that realizes the filter layout
    """
    def __init__(self, uploader):
        self.uploader = uploader # we need the uploaded to see when a new file is uploaded
        self.img = self.uploader.uploaded_image
        self.new_img = self.img
        self.sigma_s = 1
        self.sigma_b = 1

        # --------- Control Widgets --------- #
        self.brightness_slider = widgets.IntSlider(
            value=0,
            min=-255,
            max=255,
            step=1,
            description='Adjust Brightness:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.brightness_done_btn = widgets.Button(
            value = False, 
            description = 'Done', 
            disabled = False, 
            button_style = 'Success',
            icon = 'check'
        )

        self.brightness_undo_btn = widgets.Button(
            value = False,
            description = 'Undo',
            disabled = False,
            button_style = 'Primary',
            icon = 'check'
        )

        self.negative_checker = widgets.Checkbox(
            value = False,
            description = 'Negative Image',
            disabled = False,
            indent = False
        )

        self.negative_done_btn = widgets.Button(
            value = False, 
            description = 'Done', 
            disabled = False, 
            button_style = 'Success',
            icon = 'check'
        )

        self.grayscale_checker = widgets.Checkbox(
            value = False,
            description = 'Grayscale Image',
            disabled = False,
            indent = False
        )

        self.grayscale_done_btn = widgets.Button(
            value = False, 
            description = 'Done',
            disable = False,
            button_style = 'Success',
            icon = 'check'
        )

        self.contrast_slider = widgets.FloatSlider(
            value=1,
            min=0,
            max=2,
            step=0.1,
            description='Adjust Contrast:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.contrast_done_btn = widgets.Button(
            value = False, 
            description = 'Done',
            disable = False,
            button_style = 'Success',
            icon = 'check'
        )

        self.contrast_undo_btn = widgets.Button(
            value = False,
            description = 'Undo',
            disabled = False,
            button_style = 'Primary',
            icon = 'check'
        )

        self.hue_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=180,
            step=1,
            description='Adjust Hue:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.hue_done_btn = widgets.Button(
            value = False, 
            description = 'Done',
            disable = False,
            button_style = 'Success',
            icon = 'check'
        )

        self.hue_undo_btn = widgets.Button(
            value = False,
            description = 'Undo',
            disabled = False,
            button_style = 'Primary',
            icon = 'check'
        )

        self.saturation_slider = widgets.FloatSlider(
            value=1,
            min=0,
            max=2,
            step=0.1,
            description='Adjust Saturation:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False
        )

        self.saturation_done_btn = widgets.Button(
            value = False, 
            description = 'Done',
            disable = False,
            button_style = 'Success',
            icon = 'check'
        )

        self.saturation_undo_btn = widgets.Button(
            value = False,
            description = 'Undo',
            disabled = False,
            button_style = 'Primary',
            icon = 'check'
        )

        self.bilateral_sigma_s_slider = widgets.IntSlider(
            value = 1, 
            min = 1, 
            max = 10,
            step = 1,
            description = 'Adjust sigma for Gaussian',
            disabled = False,
            continuous_update = False,
            orientation = 'horizontal',
            readout = False
        )

        self.bilateral_sigma_b_slider = widgets.IntSlider(
            value = 1, 
            min = 1, 
            max = 100,
            step = 1,
            description = 'Adjust sigma for Bilateral',
            disabled = False,
            continuous_update = False,
            orientation = 'horizontal',
            readout = False
        )

        self.bilateral_done_btn = widgets.Button(
            value = False,
            description = 'Done',
            disable = False,
            button_style = 'Success',
            icon = 'check'
        )

        self.bilateral_undo_btn = widgets.Button(
            value = False,
            description = 'Undo',
            disable = False,
            button_style = 'Primary',
            icon = 'check'
        )

        # invoke handler for the uploader
        self.uploader.uploader.observe(self.new_img_output_handler, names = 'value')

        # For each control, invoke the respective handler, the handler for the done button, 
        # the handler for the undo btn (if one exists) and
        # instatiate the output and the layout

        # Brightness
        self.brightness_slider.observe(self.brightness_slider_handler, names='value')
        self.brightness_done_btn.on_click(self.done_btn_handler)
        self.brightness_undo_btn.on_click(self.undo_btn_handler)
        self.brightness_img_output = widgets.Output()
        self.brightness_btns = widgets.HBox(children=[self.brightness_done_btn, self.brightness_undo_btn]) 
        self.brightness_layout = widgets.VBox(children=[self.brightness_slider, self.brightness_img_output, self.brightness_btns])

        # Negative
        self.negative_checker.observe(self.negative_checker_handler, names='value')
        self.negative_done_btn.on_click(self.done_btn_handler)
        self.negative_img_output = widgets.Output()
        self.negative_btns = widgets.HBox(children=[self.negative_done_btn])
        self.negative_layout = widgets.VBox(children=[self.negative_checker, self.negative_img_output, self.negative_btns])

        # Grayscale
        self.grayscale_checker.observe(self.grayscale_checker_handler, names='value')
        self.grayscale_done_btn.on_click(self.done_btn_handler)
        self.grayscale_img_output = widgets.Output()
        self.grayscale_btns = widgets.HBox(children=[self.grayscale_done_btn])
        self.grayscale_layout = widgets.VBox(children=[self.grayscale_checker, self.grayscale_img_output, self.grayscale_btns])

        # Constrast
        self.contrast_slider.observe(self.contrast_slider_handler, names='value')
        self.contrast_done_btn.on_click(self.done_btn_handler)
        self.contrast_undo_btn.on_click(self.undo_btn_handler)
        self.contrast_img_output = widgets.Output()
        self.contrast_btns = widgets.HBox(children=[self.contrast_done_btn, self.contrast_undo_btn])
        self.contrast_layout = widgets.VBox(children=[self.contrast_slider, self.contrast_img_output, self.contrast_btns])

        # Hue
        self.hue_slider.observe(self.hue_slider_handler, names='value')
        self.hue_done_btn.on_click(self.done_btn_handler)
        self.hue_undo_btn.on_click(self.undo_btn_handler)
        self.hue_img_output = widgets.Output()
        self.hue_btns = widgets.HBox(children=[self.hue_done_btn, self.hue_undo_btn])
        self.hue_layout = widgets.VBox(children=[self.hue_slider, self.hue_img_output, self.hue_btns])

        # Saturation
        self.saturation_slider.observe(self.saturation_slider_handler, names='value')
        self.saturation_done_btn.on_click(self.done_btn_handler)
        self.saturation_undo_btn.on_click(self.undo_btn_handler)
        self.saturation_img_output = widgets.Output()
        self.saturation_btns = widgets.HBox(children=[self.saturation_done_btn, self.saturation_undo_btn])
        self.saturation_layout = widgets.VBox(children=[self.saturation_slider, self.saturation_img_output, self.saturation_btns])

        # Bilateral
        self.bilateral_sigma_s_slider.observe(self.bilateral_sigma_s_slider_handler, names = 'value')
        self.bilateral_sigma_b_slider.observe(self.bilateral_sigma_b_slider_handler, names = 'value')
        self.bilateral_undo_btn.on_click(self.undo_btn_handler)
        self.bilateral_done_btn.on_click(self.done_btn_handler)
        self.bilateral_img_output = widgets.Output()
        self.bilateral_btns = widgets.HBox(children=[self.bilateral_done_btn, self.bilateral_undo_btn])
        self.bilateral_sliders = widgets.VBox(children=[self.bilateral_sigma_s_slider, self.bilateral_sigma_b_slider])
        self.bilateral_layout = widgets.VBox(children=[self.bilateral_sliders, self.bilateral_img_output, self.bilateral_btns])


        # list of outputs, controls done and undo buttons
        self.outputs = [self.brightness_img_output, self.negative_img_output, self.grayscale_img_output, self.contrast_img_output, 
                        self.hue_img_output, self.saturation_img_output, self.bilateral_img_output]
        self.controls = [self.brightness_slider, self.negative_checker, self.grayscale_checker, self.contrast_slider, 
                         self.hue_slider, self.saturation_slider, self.bilateral_sigma_b_slider, self.bilateral_sigma_s_slider]
        self.done_btns = [self.brightness_done_btn, self.negative_done_btn, self.grayscale_done_btn, self.contrast_done_btn, 
                          self.hue_done_btn, self.saturation_done_btn, self.bilateral_done_btn]
        self.undo_btns = [self.brightness_undo_btn, self.contrast_undo_btn, self.hue_undo_btn, self.saturation_undo_btn, self.bilateral_undo_btn]
        
        # initialize the controls and the buttons to hidden (an image might have not been uploaded yet, an image is not the correct type)
        self.hide_items(self.controls)
        self.hide_items(self.done_btns)
        self.hide_items(self.undo_btns)

        # update all outputs with the new image when one done or undo button is clicked
        for button in self.done_btns:
            button.on_click(self.update_image_outputs_handler)

        for button in self.undo_btns:
            button.on_click(self.update_image_outputs_handler)
        
        self.layout = widgets.Accordion(children=[self.brightness_layout, self.negative_layout, self.grayscale_layout, self.contrast_layout, 
                                                  self.hue_layout, self.saturation_layout, self.bilateral_layout], 
                                        titles=['Brightness', 'Negative', 'Grayscale', 'Contrast', 'Hue', 'Saturation', 'Smoothing w/ Bilateral Filter'])

    def new_img_output_handler(self, change):
        """
        Handler for the uploaded
        """
        self.uploaded_file = self.uploader.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']
        
        if self.uploader.uploaded_image is not None and self.uploaded_file_type in cfg.supported_types_list:
            # display controls if the image type is supported
            self.img = self.new_img = self.uploader.uploaded_image
            self.display_items(self.controls)
            self.display_items(self.done_btns)
            self.display_items(self.undo_btns)
        else:
            # if wrong type is uploaded hide the buttons and the controls
            self.img = self.new_img = None
            self.hide_items(self.controls)
            self.hide_items(self.done_btns)
            self.hide_items(self.undo_btns)
        
        for output in self.outputs:
            with output:
                if self.img is not None:
                    output.clear_output(wait=True)
                    img_plot = plt.imshow(self.img)
                    img_plot = plt.axis('off')
                    plt.show()
                else:
                    output.clear_output(wait=False)
        self.reset()


    def update_image_outputs_handler(self, btn):
        """
        Handler that updates all canvases with the correct image
        when the done or undo image is clicked
        """
        for output in self.outputs:
            with output:
                if self.uploader.uploaded_image is not None:
                    output.clear_output(wait=True)
                    img_plot = plt.imshow(self.img)
                    img_plot = plt.axis('off')
                    plt.show()

    def done_btn_handler(self, btn):
        """
        Handler for the done buttons, update the image
        """
        self.new_img = self.tmp_img
        self.img = self.tmp_img
    
    def undo_btn_handler(self, btn):
        """
        Handler for the done buttons, update the image
        """
        self.img = self.new_img

        self.reset()

    def brightness_slider_handler(self, change):
        """
        Handler for the brightness slider, call the adjustBrightness function with the new value
        """
        self.brightness_img_output.clear_output(wait=True)
        self.tmp_img = adjustBrightness(change['new'], self.new_img)
        with self.brightness_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def negative_checker_handler(self, change):
        """
        Handler for the negative checker, if checked, then negative image, if unchecked, the original image
        """
        self.negative_img_output.clear_output(wait=True)
        self.tmp_img = negativeImage(change['new'], self.new_img)
        with self.negative_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def grayscale_checker_handler(self, change):
        """
        Handler for the grayscale checker, if checked, then grayscale image, if unchecked, the original image
        """
        self.grayscale_img_output.clear_output(wait=True)
        self.tmp_img = grayscaleImage(change['new'], self.new_img)
        with self.grayscale_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def contrast_slider_handler(self, change):
        """
        Handler for the contrast slider, call the adjustContrast function with the new value
        """
        self.contrast_img_output.clear_output(wait=True)
        self.tmp_img = adjustContrast(change['new'], self.new_img)
        with self.contrast_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def hue_slider_handler(self, change):
        """
        Handler for the hue slider, call the changeHue function with the new value
        """
        self.hue_img_output.clear_output(wait=True)
        self.tmp_img = changeHue(change['new'], self.new_img)
        with self.hue_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def saturation_slider_handler(self, change):
        """
        Handler for the saturation slider, call the changeSaturation function with the new value
        """
        self.saturation_img_output.clear_output(wait=True)
        self.tmp_img = changeSaturation(change['new'], self.new_img)
        with self.saturation_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def bilateral_sigma_s_slider_handler(self, change):
        """
        Handler for the sigma_s parameter of the bilater filter
        """
        self.bilateral_img_output.clear_output(wait=True)
        self.sigma_s = change['new']
        self.tmp_img = bilateralFilter(self.new_img, self.sigma_s, self.sigma_b)

        with self.bilateral_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()

    def bilateral_sigma_b_slider_handler(self, change):
        """
        Handler for the sigma_b parameter of the bilater filter
        """
        self.bilateral_img_output.clear_output(wait=True)
        self.sigma_b = change['new']
        self.tmp_img = bilateralFilter(self.new_img, self.sigma_s, self.sigma_b)

        with self.bilateral_img_output:
            img_plot = plt.imshow(self.tmp_img)
            img_plot = plt.axis('off')
            plt.show()


    def hide_items(self, items):
        """
        Hide the items in items
        """
        for item in items:
            item.layout.visibility = 'hidden'
    
    def display_items(self, items):
        """
        Make the items in items visible
        """
        for item in items:
            item.layout.visibility = 'visible'
    
    def reset(self):
        """
        Reset controls when a new image is uploaded
        """
        self.brightness_slider.value = 0
        self.negative_checker.value = False
        self.grayscale_checker = False
        self.contrast_slider.value = 1
        self.hue_slider.value = 0
        self.saturation_slider.value = 1
        self.bilateral_sigma_b_slider.value = 1
        self.bilateral_sigma_s_slider.value = 1

    
class uploadLayout:
    """
    Class that realizes the uploader tab
    """
    def __init__(self):
        self.uploader = widgets.FileUpload(
            accept = cfg.supported_types_str, 
            multiple = False
        )

        self.uploaded_image = None
        self.uploader.observe(self.uploader_handler, names='value')

        self.description = f'Supported image types are: {cfg.supported_types_str}'
        self.label = widgets.Label(value = self.description)
        self.error_msg = widgets.Output()

        # Description, upload, error msg / uploaded image
        self.layout = widgets.VBox(children = [self.label, self.uploader, self.error_msg])

    def uploader_handler(self, change):
        """
        Handler for the uploaded widget
        """
        self.error_msg.clear_output()
        self.uploaded_file = self.uploader.value[0]
        self.uploaded_file_type = self.uploaded_file['type']

        with self.error_msg:
            # if the uploaded file type is not supported print error msg
            if self.uploaded_file_type not in cfg.supported_types_list:
                print("Supported types are " + str(cfg.supported_types_str))
                print("Upload a new image")
                self.uploaded_image = None
            else:
                print(f'Image type ({self.uploaded_file_type}) is supported')
                self.uploaded_image = self.image_to_numpy()
                img_plot = plt.imshow(self.uploaded_image)
                img_plot = plt.axis('off')
                plt.show()
                
    def image_to_numpy(self):
        """
        Uploader content is in bytes, convert to numpy image
        """
        img = Image.open(io.BytesIO(self.uploaded_file['content'])).convert('RGB')
        red_channel = np.array(img.getchannel(0), dtype = np.uint8)
        green_channel = np.array(img.getchannel(1), dtype = np.uint8)
        blue_channel = np.array(img.getchannel(2), dtype = np.uint8)

        img = np.stack((red_channel, green_channel, blue_channel), axis=2)
        return img

class Layout:
    """
    Main Layout Class. It is a tab that instatiates the other classes
    """
    def __init__(self):
        self.build_layout()

    def build_layout(self):
        """
        Instatiates all tabs (Upload Image, Filters, Meme Maker, Save Image)
        """
        self.uploader = uploadLayout()
        self.filters = FilterLayout(self.uploader)
        self.meme_maker = MemeMakerLayout(self.uploader, self.filters)
        self.save_img = SaveImageLayout(self.uploader, self.meme_maker)

        self.layout = widgets.Tab(children = [self.uploader.layout, self.filters.layout, self.meme_maker.layout, self.save_img.layout],
                                  titles = ['Upload Image', 'Filters', 'Meme Maker', 'Save Image'])

    def display(self):
        """
        Call this function to make the layout visible
        """
        display(self.layout)

