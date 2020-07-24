/**
 * Quick-and-dirty implementation of a keyboard representation as required for
 * simple camera controls.
 * Not actually part of the rendering engine, but useful for debugging.
 */
use std::collections::HashMap;
pub use winit::event::{ VirtualKeyCode, KeyboardInput, ElementState };

/**
 * Keeps track of which keys are and are not pressed at any moment.
 * Absence of a key in the hash map indicates that it is not pressed, presence
 * allows for both and requires explicit checking of the stored boolean that
 * represents state to check it.
 */
pub struct Keyboard {
    pressed: HashMap<VirtualKeyCode, bool>
}

impl Keyboard {
    pub fn new() -> Self {
        Self { pressed: HashMap::new() }
    }

    pub fn process_input(&mut self, input: &KeyboardInput) {
        if let Some(virtual_keycode) = input.virtual_keycode {
            let state = input.state;
            self.pressed.insert(virtual_keycode, state == ElementState::Pressed);
        }
    }

    pub fn is_pressed(&self, virtual_keycode: VirtualKeyCode) -> bool {
        *self.pressed.get(&virtual_keycode).unwrap_or(&false)
    }
}