from dataclasses import dataclass
from typing import List, Union, Dict, Callable, Any


@dataclass
class Event:
    name: str
    value: object


@dataclass
class State:
    value: object
    actions: List
    context: Dict
    event: Event


class Machine:
    def __init__(self, definition):
        self._definition = definition

    def start(self):
        return State(self._definition['initial'],
                     [],
                     self._definition['context'],
                     None)

    def transition(self, from_state, event: Union[Event, str]):
        if isinstance(event, str):
            _event = Event(event, None)
        elif isinstance(event, Event):
            _event = event
        else:
            assert False

        state_def = self._definition['states'][from_state.value]
        if _event.name in state_def.get('on', {}):
            new_state_name = state_def['on'][_event.name]['target']
            new_state = self._definition['states'][new_state_name]
            new_context = dict(from_state.context)

            for action in state_def['on'][_event.name].get('actions', []):
                action(new_context, _event)

            return State(
                new_state_name,
                [],
                new_context,
                _event)
        return from_state


def assign(**kwargs: Dict[str, Callable[[], Any]]):
    def f(context, event):
        for k in kwargs:
            context[k] = kwargs[k]()
    return f


if __name__ == '__main__':
    def add_to_selected(context, event):
        context['start_press'] = event.value['x'], event.value['y']
        context['selected'].add(rect_at(event.value['x'], event.value['y']))

    def choose_selection(context, event):
        context['start_press'] = event.value['x'], event.value['y']
        context['selected'] = {rect_at(event.value['x'], event.value['y'])}

    canvas_machine = {
        'initial': 'idle',
        'context': {
            'selected': set(),
            },
        'states': {
            'idle': {
                'on': {
                    'primary_press': {'target': 'drag',
                                      'actions': [choose_selection]},
                    'shift_primary_press': {'target': 'drag',
                                            'actions': [add_to_selected]},
                    'secondary_press': {'target': 'scale'},
                    'pan_begin': {'target': 'pan'},
                    }
                },
            'drag': {
                'on': {
                    'primary_release': {'target': 'idle'}
                    }
                },
            'scale': {
                'on': {
                    'secondary_release': {'target': 'idle'}
                    }
                },
            'pan': {
                'on': {
                    'pan_end': {'target': 'idle'}
                    }
                },
            }
        }

    m = Machine(canvas_machine)
    s = m.start()
    next_s = m.transition(s, Event('primary_press', {'x': 10, 'y': 10}))
    assert next_s.value == 'drag'
    next_s = m.transition(s, 'primary_release')
    assert next_s.value == 'idle'
    print(next_s)
