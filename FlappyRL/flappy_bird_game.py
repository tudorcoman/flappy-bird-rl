
import pygame, random, time
import numpy as np 

class FlappyBirdGame: 
    def __init__(self, headless = False, use_keyboard = True):
        self.headless = headless
        self.use_keyboard = use_keyboard
        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((700, 487))
            pygame.display.set_caption("Flappy RL")
            self.background_image = pygame.image.load('background.jpg')
            #customize the bird
            self.bird_image = pygame.image.load('bird_rl.png')
            self.pipe_down = pygame.image.load('pipe_small_down.png')
            self.pipe_up = pygame.image.load('pipe_small_up.png')
            self.font = pygame.font.Font('freesansbold.ttf', 32) 
        self.x_location = 700
        self.x_location_change = 0
        self.running = True 
        self.bird_x = 50
        self.bird_y = 200
        self.bird_y_change = 0
        self.score = 0
        self.height = random.randint(-600, -450)
        self.jumping = False
        self.last_time_jump = -1000
        self.jump_interval = 300
        self.last_animation_time = 0 
        self.animation_time = 0
        self.aniimation_interval = 500 
        self.start_jump_y = 0
        self.max_jump_height = 50
        self.actions = []

    def reset(self):
        pygame.init()
        if not self.headless:
            self.screen = pygame.display.set_mode((700, 487))
            pygame.display.set_caption("Flappy RL")
            self.background_image = pygame.image.load('background.jpg')
            #customize the bird
            self.bird_image = pygame.image.load('bird_rl.png')
            self.pipe_down = pygame.image.load('pipe_small_down.png')
            self.pipe_up = pygame.image.load('pipe_small_up.png')
            self.font = pygame.font.Font('freesansbold.ttf', 32) 
        self.x_location = 700
        self.x_location_change = 0
        self.running = True 
        self.bird_x = 50
        self.bird_y = 200
        self.bird_y_change = 0
        self.score = 0
        self.height = random.randint(-600, -450)
        self.jumping = False
        self.last_time_jump = -1000
        self.jump_interval = 300
        self.last_animation_time = 0 
        self.animation_time = 0
        self.aniimation_interval = 500 
        self.start_jump_y = 0
        self.max_jump_height = 50
        self.actions = []

        # Return the initial state
        return self.get_state()

    #function to create and show the bird
    def create_bird(self):
        if not self.headless:
            self.screen.blit(self.bird_image, (self.bird_x, self.bird_y))

    #pipes 
    def create_pipes(self):
        if not self.headless:
            self.screen.blit(self.pipe_up, (self.x_location, self.height)) 
            self.screen.blit(self.pipe_down, (self.x_location, self.height + 800))

    #collisions
    def check_collision(self):             
        pipe_width = 100
        bird_width = 93
        if self.bird_x + bird_width >= self.x_location and self.bird_x <= self.x_location + pipe_width:
            if self.bird_y <= self.height + 614 or self.bird_y >= self.height + 800 - 80:
                return True

        return False

    def display_score(self):
        if not self.headless:
            text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
            self.screen.blit(text, (20, 20))

    def start_jump_action(self):
        self.last_animation_time = pygame.time.get_ticks()
        self.x_location_change = -0.2
        if not self.jumping:
            self.jumping = True
            self.start_jump_y = self.bird_y
            self.bird_y_change = -0.6

    def stop_jump_action(self):
        self.jumping = False 

    def evaluate_jump_actions(self):
        if self.jumping:
            if self.start_jump_y - self.bird_y > self.max_jump_height:
                self.jumping = False  # Stop the jump if max height is reached
            else:
                self.bird_y += self.bird_y_change
        else:
            self.bird_y_change = 0.2  # Falling down when not jumping

    def evaluate_bird_position(self):
        if self.bird_y <= 0: 
            self.bird_y = 0
        elif self.bird_y >= 320:
            self.bird_y = 320             
            pygame.quit()
            self.running = False
        self.bird_y += self.bird_y_change

    def evaluate_fall(self):
        self.animation_time = pygame.time.get_ticks()  
        if self.animation_time - self.last_animation_time > self.aniimation_interval:
            if self.bird_y_change < 0:
                self.last_animation_time = self.animation_time
                self.bird_y_change = 0.2                 

    def advance_game(self):
        self.x_location += self.x_location_change

        if self.x_location <= -10:
            self.x_location = 700
            self.score += 1
            self.height = random.randint(-600, -450)
        self.create_pipes()

    def handle_keyboard_events(self):
        actions = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # bird movement 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    actions.append(1)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    actions.append(0)
        return actions

    def run_game(self):
        while self.running and not self.headless:
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.background_image, (0,0))

            if self.use_keyboard:
                actions = self.handle_keyboard_events()
                self.step(actions)
            else:
                pygame.event.get()
                self.step(self.actions)
                self.actions = []
            if self.running:
                pygame.display.update()
        pygame.quit()

    def run_game_once(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.background_image, (0,0))

        if self.use_keyboard:
            actions = self.handle_keyboard_events()
            self.step(actions)
        else:
            pygame.event.get()
            self.step(self.actions)
            self.actions = []
        if self.running:
            pygame.display.update() 
        else:
            pygame.quit() 

    def run_game_rl(self, agent, state, batch_size, state_size):
        while self.running: 
            if not self.headless:
                self.screen.fill((0, 0, 0))
                self.screen.blit(self.background_image, (0,0))
            pygame.event.get()
            action = agent.choose_action(state)
            next_state, reward, done = self.step([action])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if self.running and not self.headless:
                pygame.display.update()
        pygame.quit()

    def run_game_thread(self, thread):
        thread.start()
        self.run_game()
        thread.join()

    def get_state(self):
        return (self.bird_y, self.bird_y_change, self.x_location - self.bird_x, self.height)
    
    def apply_action(self, action):
        if action == 1: 
            self.start_jump_action()
        else:
            self.stop_jump_action()

    def push_action(self, action):
        self.actions.append(action)

    def calculate_reward(self):
        return -1 if self.check_collision() else 0.1
    
    def step(self, actions):
        for action in actions:
            self.apply_action(action)

        self.evaluate_jump_actions()
        self.evaluate_bird_position()
        if not self.running:
            return self.get_state(), self.calculate_reward(), True
        
        self.create_bird()
        self.evaluate_fall()
        self.advance_game()
        self.display_score()

        if self.check_collision():
            self.running = False

        new_state = self.get_state()
        reward = self.calculate_reward()
        done = not self.running
        return new_state, reward, done
    
    
if __name__ == "__main__":
    game = FlappyBirdGame(headless=False)
    game.run_game()