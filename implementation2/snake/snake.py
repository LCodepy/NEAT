import random


class SnakeEngine:

    def __init__(self, map_size: int = 10) -> None:
        self.map_size = map_size

        self.snake = [(self.map_size // 2, self.map_size // 2)]

        self.direction = (1, 0)

        self.score = 0
        self.apple = self.create_apple()

        self.game_over = False

        self.collided_with_wall = False
        self.collided_with_tail = False

    def update(self, direction: tuple[int, int]) -> None:
        if self.game_over:
            return

        self.direction = direction

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        if not (0 <= new_head[0] < self.map_size and 0 <= new_head[1] < self.map_size):
            self.game_over = True
            self.collided_with_wall = True
            return

        if new_head in self.snake:
            self.game_over = True
            self.collided_with_tail = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.apple:
            self.score += 1

            free_cells = [(x, y) for x in range(self.map_size) for y in range(self.map_size) if (x, y) not in self.snake]

            if not free_cells:
                self.game_over = True
                return

            self.apple = random.choice(free_cells)
        else:
            self.snake.pop()

    def reset(self) -> None:
        self.game_over = False
        self.snake = [(self.map_size // 2, self.map_size // 2)]

        self.score = 0
        self.apple = self.create_apple()

        self.collided_with_wall = False
        self.collided_with_tail = False

    def is_wall(self, x: int, y: int) -> bool:
        return not (0 <= x < self.map_size and 0 <= y < self.map_size)

    def create_apple(self) -> tuple[int, int]:
        free_cells = [(x, y) for x in range(self.map_size) for y in range(self.map_size) if (x, y) not in self.snake]
        return random.choice(free_cells)
