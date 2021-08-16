def main():
    import PySimpleGUI as sg
    import tkinter
    import io
    from PIL import ImageTk

    from .constants import RESOLUTION
    from .icon_b64 import ICON
    from .quadratic import QuadraticWindows

    root = tkinter.Tk()
    root.withdraw()

    sg.theme('Material1')

    win_obj = QuadraticWindows()
    win_obj.start()

    bio_mandel = io.BytesIO()
    bio_julia = io.BytesIO()
    win_obj.pil_img_mandel.save(bio_mandel, format="PNG")
    win_obj.pil_img_julia.save(bio_julia, format="PNG")
    
    menu_def = [['File', ['Save Image', 'Save Configuration', 'Open Configuration']],
                ['Draw', ['Zoom in', 'Zoom out', 'Set parameters', 'Change resolution']],
                ['Rays', ['Draw external/internal ray(s)', ['Julia set', 'Connectedness locus']]],
                ['Equipotentials', ['Draw equipotential line(s)', ['Julia set', 'Connectedness locus']]],
                ['Function',['Quadratic', ['z^2 + c'],
                             'Cubic', ['z^3 - az + b', 'z^3 + b', 'z^3 - az'],
                             "Newton mapping (z - f(z)/f'(z))", ['f(z) = z^2 + c', 'f(z) = z^3 - az + b']]]]
        
    normal_layout = [
        [sg.Menu(menu_def, font=(None, 14))],
        [sg.Text('Connectedness locus', justification='center', font=('Helvetica', 15), key='mandel_title'),
         sg.Text('Julia set', justification='center', font=('Helvetica', 15), key='julia_title')],
        [sg.Graph(key="mandel", graph_bottom_left=(-3000, -3000), graph_top_right=(3000, 3000), canvas_size=(RESOLUTION, RESOLUTION), enable_events=True, drag_submits=True),
         sg.Graph(key="julia", graph_bottom_left=(-3000, -3000), graph_top_right=(3000, 3000), canvas_size=(RESOLUTION, RESOLUTION), enable_events=True, drag_submits=True)],
        [sg.Text('Placeholder for mandel location', justification='center', font=('Helvetica', 15), key='mandel_pos'),
         sg.Text('Placeholder for julia location', justification='center', font=('Helvetica', 15), key='julia_pos')]
    ]

    newton_layout = [
        [sg.Menu(menu_def, font=(None, 14))],
        [sg.Text('Julia set')],
        [sg.Image(key="julia", size=(RESOLUTION, RESOLUTION), enable_events=True)],
        [sg.Text('', key='julia_pos')]
    ]

    window = sg.Window("", layout=normal_layout, icon=ICON, resizable=True, titlebar_icon=ICON, finalize=True)

    window['mandel'].draw_image(data=bio_mandel.getvalue(), location=(-3000, 3000))
    window['julia'].draw_image(data=bio_julia.getvalue(), location=(-3000, 3000))
    window['mandel'].set_cursor('dotbox')
    window['julia'].set_cursor('dotbox')
    window['mandel_title'].expand(True)
    window['julia_title'].expand(True)
    window['mandel_pos'].expand(True)
    window['julia_pos'].expand(True)

    mouse_down = False
    current_rec = None
    
    while True:
        event, values = window.Read()
        if event is None or event == 'Exit':
            quit()
        elif event == "mandel" and not mouse_down:
            start_coords = values['mandel']
            current_rec = window["mandel"].draw_rectangle(start_coords, start_coords, line_color='red')
            mouse_down = "mandel"
            print("clicked on mandel", event, values)
        elif event == "mandel" and mouse_down == "mandel":
            window['mandel'].delete_figure(current_rec)
            current_rec = window['mandel'].draw_rectangle(start_coords, values['mandel'], line_color='red')
            print("clicked on mandel", event, values)
        elif event == "mandel+UP" and mouse_down == "mandel":
            window['mandel'].delete_figure(current_rec)
            current_rec = None
            mouse_down = False
        elif event == "julia":
            print("clicked on julia", event, values)
        elif event == "julia+UP" and mouse_down == "julia":
            print("left mouse up on julia", event, values)
        else:
            print("Unrecognised interaction: ", event, values)
        