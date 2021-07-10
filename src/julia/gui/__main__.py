if __name__ == "__main__":
    import PySimpleGUI as sg
    import os
    import base64

    from .constants import *

    sg.theme('Material1')
    
    menu_def = [['File', ['Save Image', 'Save Configuration', 'Open Configuration']],
                ['Draw', ['Zoom in', 'Zoom out', 'Set parameters', 'Change resolution']],
                ['Rays', ['Draw external/internal ray(s)', ['Julia set', 'Connectedness locus']]],
                ['Equipotentials', ['Draw equipotential line(s)', ['Julia set', 'Connectedness locus']]],
                ['Function',['Quadratic', ['z^2 + c'],
                             'Cubic', ['z^3 - az + b', 'z^3 + b', 'z^3 - az'],
                             'Newton mapping', ['z^2 + c', 'z^3 - az + b']]]]
    
    right_click_test = ['&Right', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]
    
    normal_layout = [
        [sg.Menu(menu_def, font=(None, 14))],
        [sg.Text('Connectedness locus'), sg.Text('Julia set')]
        [sg.Image(key="mandel", size=(RESOLUTION, RESOLUTION), enable_events=True),
         sg.Image(key="julia", size=(RESOLUTION, RESOLUTION), enable_events=True)],
        [sg.Text('', key='mandel_pos'), sg.Text('', key='julia_pos')]
    ]

    newton_layout = [
        [sg.Menu(menu_def, font=(None, 14))],
        [sg.Text('Julia set')],
        [sg.Image(key="julia", size=(RESOLUTION, RESOLUTION), enable_events=True)],
        [sg.Text('', key='julia_pos')]
    ]

    dirname = os.path.dirname(__file__)

    filepath = os.path.join(dirname, "assets", "threebrot.png")

    with open(filepath, "rb") as img_file:
        icon = base64.b64encode(img_file.read())
    
    window = sg.Window("", icon=icon, resizable=True, titlebar_icon=icon)
    window.Layout(normal_layout)

    while True:
        button, values = window.Read()
        if button is None or button == 'Exit':
            quit()
        print('Button = ', button)