#!/usr/bin/env python3

import pygame
import sys
import random
from enum import Enum, auto
import time

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
        self.sounds = {
            'wing': pygame.mixer.Sound('assets/audio/wing.wav'),
            'point': pygame.mixer.Sound('assets/audio/point.wav'),
            'hit': pygame.mixer.Sound('assets/audio/hit.wav'),
            'die': pygame.mixer.Sound('assets/audio/die.wav')
        }
    
    def play(self, sound_name):
        self.sounds[sound_name].play()

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
        for pipe in pipes:
            if (self.rect.colliderect(pipe.top_pipe) or 
                self.rect.colliderect(pipe.bottom_pipe)):
                return True
        return False

class Game:
    def __init__(self):
        try:
            pygame.init()
            # Initialize mixer for better sound latency
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            
            # Load sounds
            self.sounds = {
                'wing': pygame.mixer.Sound('assets/audio/wing.wav'),
                'hit': pygame.mixer.Sound('assets/audio/hit.wav'),
                'die': pygame.mixer.Sound('assets/audio/die.wav'),
                'point': pygame.mixer.Sound('assets/audio/point.wav')
            }
            
            self.reset_game()
            self.game_state = GAME_STATE_START
            self.high_score = 0
            self.last_time = pygame.time.get_ticks()
        except Exception as e:
            print(f"Error initializing game: {e}")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        # Clean up sounds before quitting
        pygame.mixer.quit()
        pygame.quit()

    def reset_game(self):
        self.bird = Bird(WINDOW_WIDTH // 3, WINDOW_HEIGHT // 2)
        self.pipes = []
        self.last_pipe_x = WINDOW_WIDTH
        self.score = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_state == GAME_STATE_START:
                        self.game_state = GAME_STATE_PLAYING
                    elif self.game_state == GAME_STATE_PLAYING:
                        self.bird.jump()
                    elif self.game_state == GAME_STATE_GAME_OVER:
                        self.reset_game()
                        self.game_state = GAME_STATE_PLAYING

    def draw_score(self, score, x, y):
        # Convert score to string and get width of all digits
        score_str = str(score)
        total_width = sum(NUMBER_SPRITES[int(d)].get_width() for d in score_str)
        
        # Center the score horizontally at x
        current_x = x - total_width // 2
        
        # Draw each digit
        for digit in score_str:
            digit_sprite = NUMBER_SPRITES[int(digit)]
            self.screen.blit(digit_sprite, (current_x, y))
            current_x += digit_sprite.get_width() + 2  # Add small spacing between digits

    def update(self):
        if self.game_state != GAME_STATE_PLAYING:
            return

        # Calculate delta time
        current_time = pygame.time.get_ticks()
        dt = (current_time - self.last_time) / 1000.0
        self.last_time = current_time

        # Generate new pipes
        if not self.pipes or self.last_pipe_x - self.pipes[-1].x >= PIPE_SPACING:
            new_pipe = Pipe(WINDOW_WIDTH)
            self.pipes.append(new_pipe)
            self.last_pipe_x = WINDOW_WIDTH

        # Update bird and pipes with delta time
        self.bird.update(dt)
        
        # Update and clean up pipes
        for pipe in self.pipes[:]:
            pipe.update()
            # Check for score
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.rect.x:
                pipe.passed = True
                self.score += 1
                self.sounds['point'].play()  # Play point sound
            
            if pipe.is_off_screen():
                self.pipes.remove(pipe)

        # Check for collisions
        if self.bird.check_collision(self.pipes):
            self.sounds['hit'].play()  # Play hit sound
            self.sounds['die'].play()  # Play die sound
            self.game_state = GAME_STATE_GAME_OVER
        elif (self.bird.rect.bottom >= WINDOW_HEIGHT or 
              self.bird.rect.top <= 0):
            self.sounds['hit'].play()  # Play hit sound for wall collision
            self.game_state = GAME_STATE_GAME_OVER

    def draw(self):
        # Draw background
        self.screen.blit(BACKGROUND, (0, 0))
        
        # Draw pipes and bird only during gameplay or game over
        if self.game_state in [GAME_STATE_PLAYING, GAME_STATE_GAME_OVER]:
            for pipe in self.pipes:
                pipe.draw(self.screen)
            self.bird.draw(self.screen)

        # Draw score during gameplay
        if self.game_state == GAME_STATE_PLAYING:
            self.draw_score(self.score, WINDOW_WIDTH//2, 50)

        # Draw start screen
        elif self.game_state == GAME_STATE_START:
            msg_x = WINDOW_WIDTH//2 - START_MESSAGE.get_width()//2
            msg_y = WINDOW_HEIGHT//2 - START_MESSAGE.get_height()//2
            self.screen.blit(START_MESSAGE, (msg_x, msg_y))

        # Draw game over screen with just the score
        elif self.game_state == GAME_STATE_GAME_OVER:
            # Center the game over sprite vertically and horizontally
            game_over_x = WINDOW_WIDTH//2 - GAME_OVER_SPRITE.get_width()//2
            game_over_y = WINDOW_HEIGHT//3

            # Draw game over text
            self.screen.blit(GAME_OVER_SPRITE, (game_over_x, game_over_y))
            
            # Draw only the current score
            score_y = game_over_y + GAME_OVER_SPRITE.get_height() + 50
            self.draw_score(self.score, WINDOW_WIDTH//2, score_y)

        pygame.display.flip()

    def run(self):
        previous_time = time.time()
        lag = 0.0
        MS_PER_UPDATE = 1.0 / self.FPS

        while True:
            current_time = time.time()
            elapsed = current_time - previous_time
            previous_time = current_time
            lag += elapsed

            self.handle_events()

            # Update game logic at a fixed time step
            while lag >= MS_PER_UPDATE:
                self.update()
                lag -= MS_PER_UPDATE

            # Render at whatever frame rate we can achieve
            self.draw()
            self.clock.tick(self.FPS)

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
        # Load configuration from JSON/YAML file
        pass

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
