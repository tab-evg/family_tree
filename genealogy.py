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

from graph import Graph, make_acyclic, calculate_rank, add_virtual_nodes, ordering, xcoordinate, ycoordinate


class Person(BoxLayout):
    name = StringProperty('Ivan Ivanov')
    sex = StringProperty('M')
    birthday = ObjectProperty(None)

    def __init__(self, data, **kwargs):
        super(Person, self).__init__(**kwargs)
        self.name = data['NAME']
        self.sex = data['SEX']
        self._famc = None
        self._fams = None

    def add_family_links(self, data, families: Dict[str, 'Family']):
        self._famc = families.get(data.get('FAMC', None), None)
        self._fams = families.get(data.get('FAMS', None), None)

    def __iter__(self):
        if self._famc is not None:
            yield self._famc
        if self._fams is not None:
            yield self._fams


class Family(BoxLayout):
    def __init__(self, data: Dict[str, Any], persons: Dict[str, Person], **kwargs):
        super(Family, self).__init__(**kwargs)
        self._husband = persons.get(data.get('HUSB', None), None)
        self._wife = persons.get(data.get('WIFE', None), None)
        children = data.get('CHIL', None)
        if children is not None:
            if type(children) is list:
                self._children_person = [persons.get(child, None) for child in children]
            else:
                self._children_person = [persons.get(children, None)]

    def __iter__(self):
        if self._husband is not None:
            yield self._husband
        if self._wife is not None:
            yield self._wife
        for child in self._children_person:
            yield child


class FamilyTreeGraph(RelativeLayout):

    def __init__(self, **kwargs):
        super(FamilyTreeGraph, self).__init__(**kwargs)
        self.persons = dict()
        self.families = dict()

    def parse_dict(self, data):
        print(data)
        self.persons = {key: Person(person) for key, person in data['INDI'].items()}
        self.families = {key: Family(fam, self.persons) for key, fam in data['FAM'].items()}

        for key, person in data['INDI'].items():
            self.persons[key].add_family_links(person, self.families)

        nodes = list(self.persons.values()) + list(self.families.values())
        edges = []
        for v in nodes:
            for u in v:
                edges.append((v, u))
        fg = Graph(edges, nodes)
        print(fg)
        make_acyclic(fg)
        ranks = calculate_rank(fg)
        virtual_nodes = add_virtual_nodes(fg, ranks)
        order = ordering(fg, ranks)

        for i, (indi, person) in enumerate(self.persons.items()):
            self.add_widget(person)
            print(f'{indi} : {person}: {person.size}')

        for i, (indi, fam) in enumerate(self.families.items()):
            self.add_widget(fam)
            print(f'{indi} : {fam}: {fam.size}')

        xcoord = xcoordinate(fg, ranks, order)
        print(f'xcoord={xcoord}')
        ycoord = ycoordinate(fg, ranks, order)
        print(f'ycoord={ycoord}')

        for i, (indi, person) in enumerate(self.persons.items()):
            person.pos = (xcoord[person], ycoord[person])
        for i, (indi, fam) in enumerate(self.families.items()):
            fam.pos = (xcoord[fam], ycoord[fam])

    def test(self):
        self.psn = Person()
        print('init psn.pos=', self.psn.pos)
        self.add_widget(self.psn)
        print('add psn.pos=', self.psn.pos)
        print(self.ids)
        print('psn.pos=', self.psn.pos)
        print('btn.pos=', self.psn.ids.btn.pos)
