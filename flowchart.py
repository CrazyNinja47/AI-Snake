# Boilerplate taken from : https://dr0id.bitbucket.io/legacy/pygame_tutorial00.html

# import the pygame module, so you can use it
import pygame
import glob
import pickle
import logger

UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)


class Storage:

    def __init__(self):
        self.data = None
        self.p1_head = None
        self.p1_tail = None

        self.p2_head = None
        self.p2_tail = None

        self.food_icon = None

        self.index = 0
        self.font = None


class Button:
    def __init__(self,screen, font, color, id, function, text):
        #print(f'Function: {function}')
        self.color = color
        self.id = id
        self.funct = function
        #print(f'Funct {self.funct}')
        self.font = font
        self.text = text
        #print(f'Text {self.text}')
        self.screen = screen

    def place_button(self, topleft_point):
        string_Len = len(self.text)
        font_height = self.font.size
        # write text
        text, text_pos = self.font.render("  " + self.text + " " ,"black",self.color,1)
        # shadow
        pygame.draw.rect(self.screen,"black", ( ((topleft_point[0] + 5), (topleft_point[1] + 5)) ,(text_pos.width,text_pos.height)))
        button = self.screen.blit(text, topleft_point)
        return button

class Page:
    def __init__(self,screen):
        self.buttons = []


### UTILITY

def draw_grid(screen, grid_size):
    spacer = 400 % grid_size[0]
    pygame.draw.rect(screen,"grey",((380,50),(400 - spacer,400 - spacer)))
    pygame.draw.rect(screen,"black",((380,50),(400 -spacer,400- spacer)),3)
    box_width = 400 // grid_size[0]
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            start_point = ( 380 +( box_width * x) , 50 + (box_width * y))
            pygame.draw.rect(screen,"black",(start_point,(box_width,box_width)),1)
    

def get_logs():
   return glob.glob('*.pkl')

def render_buttons(screen,font, label_array, topleft_point):
    result = ""
    button_list = []
    for idex, item in enumerate(label_array): 
        font_height = font.size
        button = Button(screen, font, "wheat", item[0] ,item[1], item[2])
        new_button = button.place_button((topleft_point[0], (topleft_point[1] + (idex * font_height) + (20 * idex))))
        button_list.append((new_button,button))
    return button_list

def load_pickle(jar, data_pointer):
    with open(jar, "rb") as f:
        data_pointer.data = pickle.load(f)
    f.close()

def draw_turn(screen, storage, step_num):
    grid_size = storage.data.map_size
    spacer = 400 % grid_size[0]
    box_width = 400 // grid_size[0]
    world = storage.data.steps[step_num].world

    #print(f'P1 Direction {world.p1_direction} , P2 Direction {world.p2_direction}')

    # Draw P2 Tail
    p2_tail = pygame.transform.scale(storage.p2_tail,(box_width,box_width))
    for val in world.p2_tail[1:]:
        screen.blit(p2_tail,(380 + (val[0] * box_width ) ,(50 + (box_width * val[1]))))
    # Draw P2 Head
    p2_head = pygame.transform.scale(storage.p2_head,(box_width,box_width))
    if world.p2_direction == (-1,0):
        p2_head = pygame.transform.rotate(p2_head,90)
    elif world.p2_direction == (1,0):
        p2_head = pygame.transform.rotate(p2_head,270)
    elif world.p2_direction == (0,1):
        p2_head = pygame.transform.rotate(p2_head,180)    
    screen.blit(p2_head,(380 + (world.p2_x * box_width ) ,(50 + (box_width * world.p2_y))))
    
    # Draw P1 Tail
    p1_tail = pygame.transform.scale(storage.p1_tail,(box_width,box_width))
    for val in world.p1_tail[1:]:
        screen.blit(p1_tail,(380 + (val[0] * box_width ) ,(50 + (box_width * val[1]))))
    # Draw P1 Head
    p1_head = pygame.transform.scale(storage.p1_head,(box_width,box_width))
    if world.p1_direction == (-1,0):
        p1_head = pygame.transform.rotate(p1_head,90)
    elif world.p1_direction == (1,0):
        p1_head = pygame.transform.rotate(p1_head,270)
    elif world.p1_direction == (0,1):
        p1_head = pygame.transform.rotate(p1_head,180)    
    screen.blit(p1_head,(380 + (world.p1_x * box_width ) ,(50 + (box_width * world.p1_y))))
    
    if world.food_drawn:
        food = pygame.transform.scale(storage.food_icon,(box_width,box_width))
        screen.blit(food,(380 + (world.food[0] * box_width ) ,(50 + (box_width * world.food[1]))))
    
    pygame.display.flip()



##  FILE SELECT
def file_select(screen, page: Page):
    screen.fill("dark grey",rect=None)
    font = pygame.freetype.SysFont('Comic Sans MS',20)
    logs = get_logs()
    button_queue = []
    for index, log in enumerate(logs):
        button_queue.append(( index, "function", log))
    page.buttons = render_buttons(screen,font,button_queue,(100,100))
    pygame.display.flip()

## VISUALIZER PAGE
def visual_page(screen, page, storage):
    font = pygame.freetype.SysFont('Comic Sans MS',20)
    go_back = Button(screen, font, "white", "select", file_select, "Go back")
    next_pg = Button(screen, font, "white", "next_pg", next_step, "Next")
    prev_pg = Button(screen, font, "white", "prev_pg", prev_step, "Previous")
    to_visual = Button(screen, font , "white", "visual", visual_page, "Back to visualizer")
    screen.fill("light blue")
    draw_grid(screen,storage.data.map_size)
    update_heuristic(screen, page, storage, storage.index)

    page.buttons = [(go_back.place_button((20,20)),go_back),
    (prev_pg.place_button((450,450)),prev_pg),
    (next_pg.place_button((650,450)),next_pg)]
    pygame.display.flip()

def view_heuristic(screen, page, storage, step):
    screen.fill("light blue")
    pygame.draw.rect(screen,"black", ((24,404) ,(200,150)))
    pygame.draw.rect(screen,"grey", ((20,400) ,(200,150)))



def update_heuristic(screen,page,storage, step):
    text, text_pos = storage.font.render("Player 2" ,"black","grey",1)
    text2, text_pos2 = storage.font.render("Heuristic" ,"black","grey",1)
    p2_choice = storage.data.steps[step].p2_tree.choice
    texta, text_posa = storage.font.render(p2_choice,"black","grey",1)

    P2_step = storage.data.steps[step].p2_tree
    if P2_step.heuristic:
        p2_val = str(round(sum(P2_step.heuristic.values()),4))
        print(f'P2 Choice: {P2_step.choice}')
        print(f'P2 Loc: ({storage.data.steps[step].world.p2_x},{storage.data.steps[step].world.p2_y})')
        print(f'Food Loc: ({storage.data.steps[step].world.food})')
        if P2_step.left_child:
            print(f'\tLeft Heuristic ({str(round(sum(P2_step.left_child.heuristic.values()),4))}): {P2_step.left_child.heuristic}')
        if P2_step.center_child:
            print(f'\tCenter Heuristic ({str(round(sum(P2_step.center_child.heuristic.values()),4))}): {P2_step.center_child.heuristic}')
        if P2_step.right_child:
            print(f'\tRight Heuristic: ({str(round(sum(P2_step.right_child.heuristic.values()),4))}){P2_step.right_child.heuristic}')

    
    else:
        p2_val = "NO HEURISTIC?"
    text6, text_pos6 = storage.font.render(p2_val,"black","grey",1)
    # shadow    
    pygame.draw.rect(screen,"black", ((24,404) ,(200,150)))
    pygame.draw.rect(screen,"grey", ((20,400) ,(200,150)))
    screen.blit(text6, (20,490))
    screen.blit(text2, (20,430))
    screen.blit(text, (20,400))
    screen.blit(texta, (20,460))



    text3, text_pos3 = storage.font.render("Player 1" ,"black","grey",1)
    text4, text_pos4 = storage.font.render("Heuristic" ,"black","grey",1)
    P1_step = storage.data.steps[step].p1_tree
    if P1_step.heuristic:
        p1_val = str(round(sum(P1_step.heuristic.values()),4))
        print(f'P1 Choice: {P1_step.choice} and facing {storage.data.steps[step].world.p1_direction}')
        print(f'P1 Loc: ({storage.data.steps[step].world.p1_x},{storage.data.steps[step].world.p1_y})')
        print(f'Food Loc: ({storage.data.steps[step].world.food})')
        if P1_step.left_child:
            print(f'\tLeft Heuristic ({str(round(sum(P1_step.left_child.heuristic.values()),4))}): {P1_step.left_child.heuristic}')
        if P1_step.center_child:
            print(f'\tCenter Heuristic ({str(round(sum(P1_step.center_child.heuristic.values()),4))}): {P1_step.center_child.heuristic}')
        if P1_step.right_child:
            print(f'\tRight Heuristic: ({str(round(sum(P1_step.right_child.heuristic.values()),4))}){P1_step.right_child.heuristic}')
        print("\n---------")

    else:
        p1_val = "NO HEURISTIC?"
    text5, text_pos5 = storage.font.render(p1_val,"black","grey",1)
    # shadow    
    textb, text_posb = storage.font.render(P1_step.choice,"black","grey",1)

    pygame.draw.rect(screen,"black", ((24,204) ,(200,150)))
    pygame.draw.rect(screen,"grey", ((20,200) ,(200,150)))
    screen.blit(text5, (20,290))
    screen.blit(text4, (20,230))
    screen.blit(text3, (20,200))
    screen.blit(textb, (20,260))




def next_step(screen, page, storage):
    if storage.index < len(storage.data.steps) - 1:
        storage.index += 1
        update_heuristic(screen, page, storage, storage.index)
        draw_grid(screen, storage.data.map_size)
        draw_turn(screen, storage, storage.index)
    else:
        print(f'Too full! Storage.index {storage.index}')

def prev_step(screen, page, storage):
    if storage.index > 0:
        storage.index -= 1
        update_heuristic(screen, page, storage, storage.index)
        draw_grid(screen, storage.data.map_size)
        draw_turn(screen, storage, storage.index)
        print(f'post Storage.index {storage.index}')
    else:
        print("too empty!")




# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    # intialize font
    pygame.font.init()
    # load and set the logo
    logo = pygame.image.load("./resources/Avatar.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Visualize Program")
    
    
    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((800,600))
    screens = []

    data = Storage()

    # Load files and remove background color (pink)
    data.food_icon = pygame.image.load("./resources/Food.png")
    data.food_icon.set_colorkey((255,83,230))

    data.p1_tail = pygame.image.load("./resources/P1Body.png")
    data.p1_tail.set_colorkey((255,83,230))
    data.p1_head = pygame.image.load("./resources/P1HeadU.png")
    data.p1_head.set_colorkey((255,83,230))

    data.p2_tail = pygame.image.load("./resources/P2Body.png")
    data.p2_tail.set_colorkey((255,83,230))
    data.p2_head = pygame.image.load("./resources/P2HeadU.png")
    data.p2_head.set_colorkey((255,83,230))

    data.font = pygame.freetype.SysFont('Comic Sans MS',20)

    select = Page(screen)
    screens.append(select)
    file_select(screen, select)

    # define a variable to control the main loop
    running = True
     
    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check all the pages we have 
                for page in screens:
                    # Check all buttons per page
                    for rect in page.buttons:
                        # If that button was pressed
                        if rect[1].id == 99 and rect[0].collidepoint(event.pos):
                            print("I forgot")
                            visual_page(screen,select)
                        elif rect[1].id == "select"  and rect[0].collidepoint(event.pos):
                            rect[1].funct(screen,select)
                        elif rect[1].id == "next_pg"  and rect[0].collidepoint(event.pos):
                            rect[1].funct(screen,select,data)
                        elif rect[1].id == "prev_pg"  and rect[0].collidepoint(event.pos):
                            rect[1].funct(screen,select,data)
                        elif rect[0].collidepoint(event.pos):
                            #print(f' {rect[1].text} {rect[1].id} Button pressed! Loading {rect[1].funct}')
                            load_pickle(rect[1].text, data)
                            visual_page(screen,select, data)
                            draw_turn(screen, data, 0)
                            data.index = 0
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
    
