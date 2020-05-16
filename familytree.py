import os

import yaml
from kivy.app import App
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import (
    ObjectProperty, StringProperty
)


class LoadDialog(FloatLayout):
    path = StringProperty(None)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def __init__(self, path, **kwargs):
        self.path = path
        super(LoadDialog, self).__init__(**kwargs)


class FamilyTree(BoxLayout):
    def test(self):
        print('pos=', self.ids.family_tree.pos)
        print('size=', self.ids.family_tree.size)
        self.ids.family_tree.test()

    def load_yaml(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.ids.family_tree.parse_dict(yaml.load(stream))


class FamilyTreeApp(App):
    def __init__(self, **kwargs):
        super(FamilyTreeApp, self).__init__(**kwargs)

    def build(self):
        self.ft = FamilyTree()
        return self.ft

    def open_file(self):
        content = LoadDialog(os.getcwd(), load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, path, filename):
        self.ft.load_yaml(path, filename)
        self.dismiss_popup()


Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    FamilyTreeApp().run()
