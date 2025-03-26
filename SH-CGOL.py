import pyglet
from pyglet.window import Window, key
import math

# Window setup
window = Window(width=640, height=480, caption="Conway Roguelite")

# Camera position for scrolling
camera_x = 0.0
camera_y = 0.0
scroll_speed = 5.0  # Adjust as needed

# Hex grid parameters
hex_size = 20  # Radius of hex (distance from center to vertex)
hex_width = hex_size * 2
hex_height = hex_size * math.sqrt(3)

# Vaporwave color palette (RGB, 0-255 range)
colors = [
    (255, 113, 206),  # Neon pink
    (1, 205, 254),    # Neon blue
    (185, 103, 255),  # Neon purple
]

# Key state tracking
keys = key.KeyStateHandler()
window.push_handlers(keys)

# Precompute hex vertices (for a single hex centered at 0,0)
hex_vertices = []
for i in range(6):
    angle = math.pi / 3 * i  # 60 degrees apart
    x = hex_size * math.cos(angle)
    y = hex_size * math.sin(angle)
    hex_vertices.extend([x, y])

# Create a batch for efficient rendering
batch = pyglet.graphics.Batch()
vertex_lists = []  # To store vertex lists for updating

# Get the default shader program
shader_program = pyglet.graphics.get_default_shader()

# Define vertex attributes for the hex grid, matching the default shader
attributes = {
    'position': {
        'location': shader_program.attributes['position']['location'],
        'format': 'f',
        'count': 2,
        'normalize': False,  # Floats don't need normalization
        'instance': False    # Not instanced
    },
    'colors': {
        'location': shader_program.attributes['colors']['location'],
        'format': 'B',
        'count': 3,
        'normalize': True,   # Colors need to be normalized (0-255 to 0.0-1.0)
        'instance': False    # Not instanced
    }
}

# Create a group with the shader program
group = pyglet.graphics.ShaderGroup(shader_program, order=0, parent=None)

# Get a vertex domain from the batch
domain = batch.get_domain(
    indexed=False,      # No indices for a line loop
    instanced=False,    # No instancing
    mode=pyglet.gl.GL_LINE_LOOP,
    group=group,       # Use the shader group
    attributes=attributes
)

def update(dt):
    """Update camera position based on key presses."""
    global camera_x, camera_y
    dx = 0
    dy = 0
    if keys[key.W]:  # Up
        dy += scroll_speed
    if keys[key.S]:  # Down
        dy -= scroll_speed
    if keys[key.A]:  # Left
        dx -= scroll_speed
    if keys[key.D]:  # Right
        dx += scroll_speed
    
    # Apply movement (diagonals work naturally with combined dx, dy)
    camera_x += dx
    camera_y += dy

@window.event
def on_draw():
    global vertex_lists
    window.clear()
    
    # Calculate visible grid bounds (with some buffer)
    grid_cols = int(window.width / (hex_width * 0.75)) + 2
    grid_rows = int(window.height / hex_height) + 2
    start_col = int(camera_x // (hex_width * 0.75)) - grid_cols // 2
    start_row = int(camera_y // hex_height) - grid_rows // 2
    
    # Clear previous vertex lists
    for vlist in vertex_lists:
        vlist.delete()
    vertex_lists.clear()
    
    # Draw hex grid
    for row in range(start_row, start_row + grid_rows):
        for col in range(start_col, start_col + grid_cols):
            # Hex center position (offset every other row)
            x = col * hex_width * 0.75
            y = row * hex_height + (hex_height / 2 if col % 2 else 0)
            
            # Adjust for camera
            screen_x = x - camera_x + window.width / 2
            screen_y = y - camera_y + window.height / 2
            
            # Skip if off-screen (optional optimization)
            if screen_x < -hex_width or screen_x > window.width + hex_width or \
               screen_y < -hex_height or screen_y > window.height + hex_height:
                continue
            
            # Apply color from palette (cycle through colors)
            color = colors[(row + col) % len(colors)]
            
            # Create vertex list for this hex
            vertices = []
            for i in range(0, len(hex_vertices), 2):
                vertices.extend([screen_x + hex_vertices[i], screen_y + hex_vertices[i + 1]])
            
            # Create a vertex list in the domain
            vlist = domain.create(6)  # 6 vertices for a hex
            vlist.position[:] = vertices
            vlist.colors[:] = color * 6  # Repeat color for each vertex
            vertex_lists.append(vlist)
    
    # Draw the batch
    batch.draw()

# Schedule updates at 60 FPS
pyglet.clock.schedule_interval(update, 1/60.0)

# Run the application
pyglet.app.run()