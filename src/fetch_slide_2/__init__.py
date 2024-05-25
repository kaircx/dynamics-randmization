from gymnasium import register

register(
    id='FetchSlide2-v1',
    entry_point='fetch_slide_2.envs:FetchSlide2',
    max_episode_steps=50,
)
print("registered FetchSlide2-v1 environment")
