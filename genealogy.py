from typing import Any, Dict

from kivy import utils
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget


class Person(BoxLayout):
    name = StringProperty('Ivan Ivanov')
    sex = StringProperty('M')
    birthday = ObjectProperty(None)

    def __init__(self, data, **kwargs):
        super(Person, self).__init__(**kwargs)
        self.name = data['NAME']
        self.sex = data['SEX']


class Family(BoxLayout):
    def __init__(self, data: Dict[str, Any], persons: Dict[str, Person], **kwargs):
        super(Family, self).__init__(**kwargs)
        self._husband = persons.get(data['HUSB'], None)
        self._wife = persons.get(data['WIFE'], None)
        self._children_person = [persons.get(child, None) for child in data['CHIL']]


class FamilyTreeGraph(RelativeLayout):

    def __init__(self, **kwargs):
        super(FamilyTreeGraph, self).__init__(**kwargs)
        self.persons = dict()
        self.families = dict()

    def parse_dict(self, data):
        print(data)
        self.persons = {key: Person(person) for key, person in data['INDI'].items()}
        self.families = {key: Family(fam, self.persons) for key, fam in data['FAM'].items()}

        for i, (indi, person) in enumerate(self.persons.items()):
            print(f'{indi} : {person}')
            self.add_widget(person)
            person.pos = (0, 100 * i)
        for i, (indi, fam) in enumerate(self.families.items()):
            print(f'{indi} : {fam}')
            self.add_widget(fam)
            fam.pos = (400, 100 * i)

    def test(self):
        self.psn = Person()
        print('init psn.pos=', self.psn.pos)
        self.add_widget(self.psn)
        print('add psn.pos=', self.psn.pos)
        print(self.ids)
        print('psn.pos=', self.psn.pos)
        print('btn.pos=', self.psn.ids.btn.pos)
