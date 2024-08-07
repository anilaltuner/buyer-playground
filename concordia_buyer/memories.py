from concordia import typing
from concordia.associative_memory import associative_memory


class Memories(typing.component.Component):
    """Component that displays recently written memories."""

    def __init__(
            self,
            memory: associative_memory.AssociativeMemory,
            component_name: str = 'memories',
    ):
        """Initializes the component.

    Args:
      memory: Associative memory to add and retrieve observations.
      component_name: Name of this component.
    """
        self._name = component_name
        self._memory = memory

    def name(self) -> str:
        return self._name

    def state(self):
        # Retrieve up to 1000 of the latest memories.
        memories = self._memory.retrieve_recent(k=1000, add_time=True)
        # Concatenate all retrieved memories into a single string and put newline
        # characters ("\n") between each memory.
        return '\n'.join(memories) + '\n'

    def get_last_log(self):
        return {
            'Summary': 'observation',
            'state': self.state().splitlines(),
        }
