#!/usr/bin/env python3

import pygame
import sys
import random
from enum import Enum, auto
import time
import yaml
import numpy as np
import pygame.mixer as mixer
from dataclasses import dataclass
from typing import Dict, List, Set, TypeVar, Generic, Optional, Any
from collections import defaultdict, deque
import asyncio
import psutil
from contextlib import contextmanager

# Initialize Pygame
pygame.init()

# Constants
SCALE = 1.5  # Adjust this to make the window bigger/smaller
WINDOW_WIDTH = int(288 * SCALE)  # Original width * scale
WINDOW_HEIGHT = int(512 * SCALE)  # Original height * scale
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game dimensions
BIRD_WIDTH = int(34 * SCALE)
BIRD_HEIGHT = int(24 * SCALE)
PIPE_WIDTH = int(80 * SCALE)
PIPE_GAP = int(200 * SCALE)
PIPE_SPACING = int(400 * SCALE)
PIPE_SPEED = int(3 * SCALE)
MIN_PIPE_HEIGHT = int(50 * SCALE)

# Physics
GRAVITY = 0.5 * SCALE
JUMP_STRENGTH = -10 * SCALE

# Load and scale images
BACKGROUND = pygame.transform.scale(
    pygame.image.load("assets/sprites/background-day.png"), 
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

# Load and scale pipe sprite
PIPE_SPRITE = pygame.image.load("assets/sprites/pipe-green.png")
PIPE_SPRITE = pygame.transform.scale(
    PIPE_SPRITE,
    (PIPE_WIDTH, int(PIPE_SPRITE.get_height() * SCALE))
)

# Load and scale UI elements
GAME_OVER_SPRITE = pygame.transform.scale(
    pygame.image.load("assets/sprites/gameover.png"),
    (int(pygame.image.load("assets/sprites/gameover.png").get_width() * SCALE),
     int(pygame.image.load("assets/sprites/gameover.png").get_height() * SCALE))
)

START_MESSAGE = pygame.transform.scale(
    pygame.image.load("assets/sprites/message.png"),
    (int(pygame.image.load("assets/sprites/message.png").get_width() * SCALE),
     int(pygame.image.load("assets/sprites/message.png").get_height() * SCALE))
)

# Load and scale bird sprites
BIRD_SPRITES = [
    pygame.transform.scale(
        pygame.image.load(f"assets/sprites/bluebird-{flap}flap.png"),
        (BIRD_WIDTH, BIRD_HEIGHT)
    )
    for flap in ['down', 'mid', 'up']
]

# Load and scale number sprites
NUMBER_SPRITES = [
    pygame.transform.scale(
        pygame.image.load(f"assets/sprites/{i}.png"),
        (int(pygame.image.load(f"assets/sprites/{i}.png").get_width() * SCALE),
         int(pygame.image.load(f"assets/sprites/{i}.png").get_height() * SCALE))
    )
    for i in range(10)
]

# Game states
GAME_STATE_START = 0
GAME_STATE_PLAYING = 1
GAME_STATE_GAME_OVER = 2

# Animation
ANIMATION_SPEED = 0.15  # Time between frames in seconds

class SoundManager:
    def __init__(self):
        mixer.pre_init(44100, -16, 2, 512)
        mixer.init()
        self.sounds = {}
        self.music = {}
        
    def load_sound(self, name, path):
        self.sounds[name] = mixer.Sound(path)
        
    def load_music(self, name, path):
        self.music[name] = path
        
    def play_sound(self, name, volume=1.0):
        if name in self.sounds:
            self.sounds[name].set_volume(volume)
            self.sounds[name].play()

class Pipe:
    def __init__(self, x):
        self.x = x
        # Randomly generate the gap position
        gap_y = random.randint(MIN_PIPE_HEIGHT + PIPE_GAP//2, 
                             WINDOW_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP//2)
        
        self.top_height = gap_y - PIPE_GAP//2
        self.bottom_y = gap_y + PIPE_GAP//2
        
        # Create rectangles for collision
        self.top_pipe = pygame.Rect(x, 0, PIPE_WIDTH, self.top_height)
        self.bottom_pipe = pygame.Rect(x, self.bottom_y, PIPE_WIDTH, 
                                     WINDOW_HEIGHT - self.bottom_y)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

    def draw(self, screen):
        # Draw top pipe (flipped)
        top_pipe = pygame.transform.flip(PIPE_SPRITE, False, True)
        screen.blit(top_pipe, (self.x, self.top_height - PIPE_SPRITE.get_height()))
        
        # Draw bottom pipe
        screen.blit(PIPE_SPRITE, (self.x, self.bottom_y))

    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0

class Bird:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, BIRD_WIDTH, BIRD_HEIGHT)
        self.velocity = 0
        self.animation_index = 0
        self.animation_timer = 0
        self.sprites = BIRD_SPRITES
    
    def animate(self, dt):
        # Update animation timer
        self.animation_timer += dt
        if self.animation_timer >= ANIMATION_SPEED:
            # Move to next frame
            self.animation_index = (self.animation_index + 1) % len(self.sprites)
            self.animation_timer = 0
    
    def jump(self):
        self.velocity = JUMP_STRENGTH
        # Play wing sound when flapping
        pygame.mixer.Sound('assets/audio/wing.wav').play()
    
    def update(self, dt):
        # Apply gravity
        self.velocity += GRAVITY
        # Update position
        self.rect.y += self.velocity
        
        # Keep the bird within the screen bounds
        if self.rect.bottom > WINDOW_HEIGHT:
            self.rect.bottom = WINDOW_HEIGHT
            self.velocity = 0
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0
            
        # Update animation
        self.animate(dt)
    
    def draw(self, screen):
        # Get current sprite
        current_sprite = self.sprites[self.animation_index]
        
        # Calculate rotation based on velocity
        # Clamp rotation between -90 and 45 degrees
        rotation = max(-90, min(45, self.velocity * 3))
        
        # Rotate sprite
        rotated_sprite = pygame.transform.rotate(current_sprite, rotation)
        
        # Get new rect for rotated sprite (keeps center position)
        rotated_rect = rotated_sprite.get_rect(center=self.rect.center)
        
        # Draw the rotated sprite
        screen.blit(rotated_sprite, rotated_rect)

    def check_collision(self, pipes):
        # Convert bird corners to numpy array
        bird_corners = np.array([
            [self.rect.left, self.rect.top],
            [self.rect.right, self.rect.top],
            [self.rect.left, self.rect.bottom],
            [self.rect.right, self.rect.bottom]
        ])
        
        for pipe in pipes:
            # Create arrays for top and bottom pipe separately
            top_pipe_rect = np.array([
                pipe.top_pipe.left,
                pipe.top_pipe.top,
                pipe.top_pipe.right,
                pipe.top_pipe.bottom
            ])
            
            bottom_pipe_rect = np.array([
                pipe.bottom_pipe.left,
                pipe.bottom_pipe.top,
                pipe.bottom_pipe.right,
                pipe.bottom_pipe.bottom
            ])
            
            # Check collision for each corner against each pipe
            for corner in bird_corners:
                x, y = corner
                # Check if point is inside top pipe
                if (x >= top_pipe_rect[0] and x <= top_pipe_rect[2] and
                    y >= top_pipe_rect[1] and y <= top_pipe_rect[3]):
                    return True
                
                # Check if point is inside bottom pipe
                if (x >= bottom_pipe_rect[0] and x <= bottom_pipe_rect[2] and
                    y >= bottom_pipe_rect[1] and y <= bottom_pipe_rect[3]):
                    return True
        
        return False

class Game:
    def __init__(self):
        try:
            pygame.init()
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            
            # Initialize systems
            self.world = World()
            self.event_manager = EventManager()
            self.resource_manager = ResourceManager()
            self.profiler = Profiler()
            self.scene_graph = SceneNode()
            self.render_system = RenderSystem()
            
            # Game state
            self.game_state = GameState.START
            self.bird = None
            self.pipes = []
            self.score = 0
            self.high_score = 0
            
            # Initialize sounds with adjusted volumes
            self.sounds = {
                'wing': pygame.mixer.Sound('assets/audio/wing.wav'),
                'hit': pygame.mixer.Sound('assets/audio/hit.wav'),
                'die': pygame.mixer.Sound('assets/audio/die.wav'),
                'point': pygame.mixer.Sound('assets/audio/point.wav')
            }
            
            for sound in self.sounds.values():
                sound.set_volume(0.3)
            
            self.reset_game()
            
        except Exception as e:
            print(f"Error initializing game: {e}")
            self.cleanup()
            sys.exit(1)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.handle_space_press()
                    
            # Convert Pygame events to our event system
            self.event_manager.emit(Event(
                "pygame_event",
                {"event": event}
            ))

    def handle_space_press(self):
        if self.game_state == GameState.START:
            self.game_state = GameState.PLAYING
        elif self.game_state == GameState.PLAYING:
            if self.bird:
                self.bird.jump()
                self.sounds['wing'].play()
        elif self.game_state == GameState.GAME_OVER:
            self.reset_game()
            self.game_state = GameState.PLAYING

    def reset_game(self):
        # Create bird entity
        bird_entity = self.world.create_entity()
        self.bird = Bird(WINDOW_WIDTH // 3, WINDOW_HEIGHT // 2)
        
        # Add components to bird
        self.world.add_component(bird_entity, Physics(
            velocity=np.array([0.0, 0.0]),
            acceleration=np.array([0.0, GRAVITY])
        ))
        self.world.add_component(bird_entity, Sprite(
            surface=BIRD_SPRITES[0],
            rect=self.bird.rect
        ))
        self.world.add_component(bird_entity, Animation(
            frames=BIRD_SPRITES,
            frame_duration=ANIMATION_SPEED
        ))
        
        self.pipes = []
        self.score = 0

    def update(self):
        with self.profiler.profile_scope("update"):
            if self.game_state == GameState.PLAYING:
                # Update game logic
                current_time = pygame.time.get_ticks()
                dt = (current_time - getattr(self, 'last_time', current_time)) / 1000.0
                self.last_time = current_time
                
                # Update bird
                if self.bird:
                    self.bird.update(dt)
                
                # Update pipes
                self.update_pipes()
                
                # Check collisions
                if self.check_collisions():
                    self.game_state = GameState.GAME_OVER
                    self.sounds['hit'].play()
                    self.sounds['die'].play()
                
                # Update systems
                self.world.update()
                self.profiler.update()

    def update_pipes(self):
        # Generate new pipes
        if not self.pipes or self.pipes[-1].x <= WINDOW_WIDTH - PIPE_SPACING:
            self.pipes.append(Pipe(WINDOW_WIDTH))
        
        # Update and remove off-screen pipes
        for pipe in self.pipes[:]:
            pipe.update()
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
            elif not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.rect.x:
                pipe.passed = True
                self.score += 1
                self.sounds['point'].play()

    def check_collisions(self):
        if self.bird:
            return self.bird.check_collision(self.pipes)
        return False

    def draw(self):
        with self.profiler.profile_scope("render"):
            # Clear screen
            self.screen.blit(BACKGROUND, (0, 0))
            
            # Draw game elements based on state
            if self.game_state in [GameState.PLAYING, GameState.GAME_OVER]:
                # Draw pipes
                for pipe in self.pipes:
                    pipe.draw(self.screen)
                
                # Draw bird
                if self.bird:
                    self.bird.draw(self.screen)
                
                # Draw score
                self.draw_score(self.score, WINDOW_WIDTH//2, 50)
            
            # Draw state-specific overlays
            if self.game_state == GameState.START:
                self.draw_start_screen()
            elif self.game_state == GameState.GAME_OVER:
                self.draw_game_over_screen()
            
            pygame.display.flip()

    def draw_score(self, score, x, y):
        score_str = str(score)
        total_width = sum(NUMBER_SPRITES[int(d)].get_width() for d in score_str)
        current_x = x - total_width // 2
        
        for digit in score_str:
            digit_sprite = NUMBER_SPRITES[int(digit)]
            self.screen.blit(digit_sprite, (current_x, y))
            current_x += digit_sprite.get_width()

    def draw_start_screen(self):
        msg_x = WINDOW_WIDTH//2 - START_MESSAGE.get_width()//2
        msg_y = WINDOW_HEIGHT//2 - START_MESSAGE.get_height()//2
        self.screen.blit(START_MESSAGE, (msg_x, msg_y))

    def draw_game_over_screen(self):
        game_over_x = WINDOW_WIDTH//2 - GAME_OVER_SPRITE.get_width()//2
        game_over_y = WINDOW_HEIGHT//3
        self.screen.blit(GAME_OVER_SPRITE, (game_over_x, game_over_y))

    def cleanup(self):
        pygame.mixer.quit()
        pygame.quit()

class ResourceLoader:
    @staticmethod
    def load_image(path, scale_size=None):
        try:
            image = pygame.image.load(path)
            if scale_size:
                return pygame.transform.scale(image, scale_size)
            return image
        except pygame.error as e:
            print(f"Couldn't load image: {path}")
            print(e)
            sys.exit(1)
    
    @staticmethod
    def load_sprite_series(template, values, scale_size=None):
        return [ResourceLoader.load_image(template.format(val), scale_size) 
                for val in values]

class GameState(Enum):
    START = auto()
    PLAYING = auto()
    GAME_OVER = auto()

class GameStateManager:
    def __init__(self, game):
        self.game = game
        self.state = GameState.START
        self.state_handlers = {
            GameState.START: self.handle_start_state,
            GameState.PLAYING: self.handle_playing_state,
            GameState.GAME_OVER: self.handle_game_over_state
        }

    def update(self):
        self.state_handlers[self.state]()

    def handle_start_state(self):
        # Start state logic
        pass

    def handle_playing_state(self):
        # Playing state logic
        pass

    def handle_game_over_state(self):
        # Game over state logic
        pass

class Config:
    def __init__(self):
        self.SCALE = 1.5
        self.WINDOW_WIDTH = int(288 * self.SCALE)
        self.WINDOW_HEIGHT = int(512 * self.SCALE)
        self.FPS = 60
        self.GRAVITY = 0.5 * self.SCALE
        self.JUMP_STRENGTH = -10 * self.SCALE
        # ... other constants ...

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls.from_dict(config_data)

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.update_times = []
        
    def log_frame_time(self, dt):
        self.frame_times.append(dt)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def get_average_fps(self):
        if not self.frame_times:
            return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

@dataclass
class Component:
    """Base class for all components"""
    pass

@dataclass
class Physics(Component):
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: float = 1.0

@dataclass
class Sprite(Component):
    surface: pygame.Surface
    rect: pygame.Rect
    rotation: float = 0.0
    layer: int = 0

@dataclass
class Animation(Component):
    frames: List[pygame.Surface]
    frame_duration: float
    current_frame: int = 0
    time_accumulated: float = 0.0

class Entity:
    def __init__(self, entity_id: int):
        self.id = entity_id
        self.components: Dict[type, Component] = {}

class World:
    def __init__(self):
        self.entities: Dict[int, Entity] = {}
        self.systems: List[System] = []
        self.next_entity_id: int = 0
        self.component_pools: Dict[type, Set[Entity]] = {}
    
    def create_entity(self) -> Entity:
        entity = Entity(self.next_entity_id)
        self.next_entity_id += 1
        self.entities[entity.id] = entity
        return entity
    
    def add_component(self, entity: Entity, component: Component):
        component_type = type(component)
        entity.components[component_type] = component
        
        if component_type not in self.component_pools:
            self.component_pools[component_type] = set()
        self.component_pools[component_type].add(entity)
    
    def update(self):
        for system in self.systems:
            system.update(self)

class SceneNode:
    def __init__(self):
        self.children: List[SceneNode] = []
        self.parent: Optional[SceneNode] = None
        self.local_transform = np.eye(3)  # 3x3 transformation matrix
        self._world_transform = None
        
    @property
    def world_transform(self) -> np.ndarray:
        if self._world_transform is None:
            if self.parent:
                self._world_transform = np.dot(
                    self.parent.world_transform, 
                    self.local_transform
                )
            else:
                self._world_transform = self.local_transform.copy()
        return self._world_transform

class RenderSystem:
    def __init__(self):
        self.scene_root = SceneNode()
        self.render_layers: Dict[int, List[Entity]] = defaultdict(list)
        self.render_target = pygame.Surface(
            (WINDOW_WIDTH, WINDOW_HEIGHT), 
            flags=pygame.SRCALPHA | pygame.HWSURFACE | pygame.DOUBLEBUF
        )

class Event:
    def __init__(self, event_type: str, data: Dict = None):
        self.type = event_type
        self.data = data or {}
        self.timestamp = time.perf_counter()

class EventManager:
    def __init__(self):
        self.listeners: Dict[str, List[callable]] = defaultdict(list)
        self.event_queue = deque(maxlen=1000)
        self.event_history = np.zeros((1000,), dtype=[
            ('type', 'U32'),
            ('timestamp', 'f8'),
            ('entity_id', 'i4')
        ])
        self.history_index = 0
        
    def emit(self, event: Event):
        self.event_queue.append(event)
        self._record_event(event)
        
    def _record_event(self, event: Event):
        self.event_history[self.history_index] = (
            event.type,
            event.timestamp,
            event.data.get('entity_id', -1)
        )
        self.history_index = (self.history_index + 1) % 1000

class ResourceManager:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.loading_tasks: List[asyncio.Task] = []
        self.preload_queue = asyncio.Queue()
        
    async def preload_resources(self, paths: List[str]):
        for path in paths:
            await self.preload_queue.put(path)
            
    async def _load_resource(self, path: str) -> Any:
        if path.endswith(('.png', '.jpg')):
            return await asyncio.to_thread(
                lambda: pygame.image.load(path).convert_alpha()
            )
        elif path.endswith('.wav'):
            return await asyncio.to_thread(
                lambda: pygame.mixer.Sound(path)
            )

class Profiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_usage = []
        self.process = psutil.Process()
        
    @contextmanager
    def profile_scope(self, name: str):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            duration = time.perf_counter_ns() - start
            self.timings[name].append(duration / 1e6)  # Convert to milliseconds
            
    def update(self):
        self.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)
        if len(self.memory_usage) > 1000:
            self.memory_usage.pop(0)

def main():
    pygame.init()
    game = Game()
    
    while True:
        game.handle_events()
        game.update()
        game.draw()
        game.clock.tick(FPS)

if __name__ == "__main__":
    main()
