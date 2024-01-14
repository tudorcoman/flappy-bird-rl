import random
from turtledemo import clock

import pygame
import random

pygame.init()

#game display and background
screen = pygame.display.set_mode((700, 487))
pygame.display.set_caption("Flappy RL")
background_image = pygame.image.load('background.jpg')

#customize the bird
bird_image = pygame.image.load('bird_rl.png')
bird_x = 50
bird_y = 200
bird_y_change = 0
score = 0
#function to create and show the bird
def create_bird(x, y):
    screen.blit(bird_image, (x, y))

pipe_down = pygame.image.load('pipe_small_down.png')
pipe_up = pygame.image.load('pipe_small_up.png')
font = pygame.font.Font('freesansbold.ttf', 32) 
def display_score():
    text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(text, (20, 20))
#pipes
x_location = 700
x_location_change = 0
height = random.randint(-600, -450)

def create_pipe(height):
    screen.blit(pipe_up, (x_location, height)) 
    screen.blit(pipe_down, (x_location, height + 800))

#colisions
def check_collision(x_location, height, bird_y):
    # print(x_location)                  
    pipe_width = 100
    bird_width = 93
    bird_x = 50
    if bird_x + bird_width >= x_location and bird_x <= x_location + pipe_width:
        # print("bird y", bird_y)
        # print("height", height)
        # print("height + 614", height + 614)
        # print("height + 750", height + 750)

        if bird_y <= height + 614 or bird_y >= height + 800 - 80:
            return True

    return False


running = True
jumping = False
last_time_jump = -1000
jump_interval = 300
last_animation_time = 0 
animation_time = 0
aniimation_interval = 300                   
while running:
    # display_score()
    #show backround image while game is running
    screen.fill((0, 0, 0))
    screen.blit(background_image, (0,0))
    current_time = pygame.time.get_ticks()  # get current time
    #show the quit game button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # bird movement 
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                last_time_jump = current_time
                last_animation_time = pygame.time.get_ticks()                      
                print ("am apasat space")
                bird_y_change = -0.2
                x_location_change = -0.2
                jumping = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                bird_y_change = 0.2



    #bird bounders
    if bird_y <= 0: 
        bird_y = 0
    elif bird_y >= 320:
        bird_y = 320             
        pygame.quit()
        break
    bird_y += bird_y_change
    create_bird(bird_x, bird_y)
    # wait 100 milliseconds
    animation_time = pygame.time.get_ticks()  
    if animation_time - last_animation_time > aniimation_interval:
        if bird_y_change < 0:
            print("Tre sa cad")
            last_animation_time = animation_time
            bird_y_change = 0.2                 
    # if jumping == True:
    #     bird_y_change = 0.5
            
    #pipe creation
    x_location += x_location_change

    #pipe bounders
    if x_location <= -10:
        x_location = 700
        height = random.randint(-600, -450)
        score += 1
        print(score)
    create_pipe(height)

    collision = check_collision(x_location, height, bird_y)

    if collision:
        running = False
    display_score()
    pygame.display.update()


pygame.quit()

