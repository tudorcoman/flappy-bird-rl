import pygame
from ple import PLE
from ple.games.flappybird import FlappyBird

def main():
    # Create a FlappyBird game instance
    game = FlappyBird()

    # Create a PLE environment with the FlappyBird game
    env = PLE(game, fps=30, display_screen=True)

    # Initialize the environment
    env.init()

    # Set the game to human mode
    env.getGameState = lambda: {}
    env.act(0)
    clock = pygame.time.Clock()
    is_jumping = False

    while not env.game_over():
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.game_over()

            # Check for key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not is_jumping:
                    # Make the bird jump when the space bar is pressed
                    env.act(119)  # ASCII code for 'w' key
                    is_jumping = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    is_jumping = False

        # Get the current state
        state = env.getScreenRGB()

        # Perform a no-op action if the space bar is not pressed
        if not is_jumping:
            env.act(0)

        # Step the environment
        reward = env.act(0)

        # Print the current reward
        print("Reward:", reward)

        # Control the frame rate
        clock.tick(env.fps)

    pygame.quit()

if __name__ == "__main__":
    main()