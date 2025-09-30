import os
import glob
import math
from panda3d.core import *

class BillLoader:
    def __init__(self, app):
        self.app = app
        self.render = app.render
        self.loader = app.loader
        self.taskMgr = app.taskMgr
    
    def find_bill_images(self, name, denomination):
        """Find bill images with flexible naming patterns"""
        denom_path = f"./images/{name}/{denomination}"
        
        if not os.path.exists(denom_path):
            return None, None
        
        # Get all PNG and JPG files in the directory
        all_images = glob.glob(os.path.join(denom_path, "*.png")) + glob.glob(os.path.join(denom_path, "*.jpg"))
        
        front_path = None
        back_path = None
        
        # Try to identify front and back images
        for img_path in all_images:
            filename = os.path.basename(img_path).lower()
            
            # Check for front images
            if any(pattern in filename for pattern in ["front", "_f", "f_", "f.png", "f.jpg"]):
                front_path = img_path
            # Check for back images
            elif any(pattern in filename for pattern in ["back", "_b", "b_", "b.png", "b.jpg"]):
                back_path = img_path
        
        # If we didn't find specific front/back, use first two images
        if not front_path and len(all_images) > 0:
            front_path = all_images[0]
        if not back_path and len(all_images) > 1:
            back_path = all_images[1]
        elif not back_path and len(all_images) > 0:
            back_path = all_images[0]  # Use same image for both sides if only one exists
        
        return front_path, back_path
    
    def load_bill(self, room_index, denom_index, name, denomination, pos):
        # Find bill images with flexible naming
        front_path, back_path = self.find_bill_images(name, denomination)
        
        # Check if files exist
        if not front_path or not back_path:
            print(f"Warning: Bill images not found for {name} ${denomination}")
            print(f"  Looking in: ./images/{name}/{denomination}/")
            
            # Create a placeholder bill
            self.create_placeholder_bill(name, denomination, pos)
            return
        
        try:
            # Create two planes for front and back
            card_front = self.loader.loadModel("models/plane.bam")
            card_front.setScale(1.0, 0.1, 0.5)  # Bill shape
            card_front.setPos(pos.x, pos.y, pos.z)
            card_front.setH(0)  # Face forward
            card_front.reparentTo(self.render)
            
            card_back = self.loader.loadModel("models/plane.bam")
            card_back.setScale(1.0, 0.1, 0.5)
            card_back.setPos(pos.x, pos.y, pos.z)
            card_back.setH(180)  # Face backward
            card_back.reparentTo(self.render)
            
            # Load textures
            front_tex = self.loader.loadTexture(front_path)
            back_tex = self.loader.loadTexture(back_path)
            
            # Apply textures
            card_front.setTexture(front_tex, 1)
            card_back.setTexture(back_tex, 1)
            
            # Make bills emissive (self-illuminated)
            card_front.setAttrib(LightAttrib.makeAllOff())
            card_back.setAttrib(LightAttrib.makeAllOff())
            
            # Add a slight glow effect
            card_front.setShaderAuto()
            card_back.setShaderAuto()
            
            # Add spinning animation to both
            self.taskMgr.add(self.spin_bill_task, f"spin_bill_{room_index}_{denom_index}_front", 
                            extraArgs=[card_front, denomination], appendTask=True)
            self.taskMgr.add(self.spin_bill_task, f"spin_bill_{room_index}_{denom_index}_back", 
                            extraArgs=[card_back, denomination], appendTask=True)
            
            print(f"Successfully loaded bill: {name} ${denomination}")
            
        except Exception as e:
            print(f"Error loading bill {name} ${denomination}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_placeholder_bill(self, name, denomination, pos):
        """Create a placeholder bill when images aren't found"""
        try:
            # Create a simple colored card as placeholder
            card_front = self.loader.loadModel("models/plane.bam")
            card_front.setScale(1.0, 0.1, 0.5)
            card_front.setPos(pos.x, pos.y, pos.z)
            card_front.setH(0)
            
            card_back = self.loader.loadModel("models/plane.bam")
            card_back.setScale(1.0, 0.1, 0.5)
            card_back.setPos(pos.x, pos.y, pos.z)
            card_back.setH(180)
            
            # Use a color based on denomination
            hue = (math.log10(max(denomination, 10)) / 9) % 1.0
            color = self.hsv_to_rgb(hue, 0.8, 0.8)
            card_front.setColor(color[0], color[1], color[2], 1)
            card_back.setColor(color[0], color[1], color[2], 1)
            
            card_front.reparentTo(self.render)
            card_back.reparentTo(self.render)
            
            # Add text label to front only
            text = TextNode('placeholder_bill')
            text.setText(f"{name}\n${denomination:,}")
            text.setTextColor(1, 1, 1, 1)
            text.setAlign(TextNode.ACenter)
            text_node = card_front.attachNewNode(text)
            text_node.setScale(0.2)
            text_node.setPos(0, 0.1, 0)
            
            # Add spinning animation to both
            self.taskMgr.add(self.spin_bill_task, f"spin_bill_placeholder_{name}_{denomination}_front", 
                            extraArgs=[card_front, denomination], appendTask=True)
            self.taskMgr.add(self.spin_bill_task, f"spin_bill_placeholder_{name}_{denomination}_back", 
                            extraArgs=[card_back, denomination], appendTask=True)
            
            print(f"Created placeholder bill for {name} ${denomination}")
            
        except Exception as e:
            print(f"Error creating placeholder bill for {name} ${denomination}: {e}")
            import traceback
            traceback.print_exc()
    
    def spin_bill_task(self, bill_node, denomination, task):
        # Rotate the bill slowly
        spin_speed = 10.0  # degrees per second
        bill_node.setH(bill_node.getH() + spin_speed * globalClock.getDt())
        return Task.cont
    
    def hsv_to_rgb(self, h, s, v):
        # Convert HSV color to RGB
        if s == 0.0:
            return (v, v, v)
        
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i % 6 == 0:
            return (v, t, p)
        elif i % 6 == 1:
            return (q, v, p)
        elif i % 6 == 2:
            return (p, v, t)
        elif i % 6 == 3:
            return (p, q, v)
        elif i % 6 == 4:
            return (t, p, v)
        else:
            return (v, p, q)