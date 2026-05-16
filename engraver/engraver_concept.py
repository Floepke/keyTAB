'''
# the problem
i see now that i did go the wrong direction. problems:
- the partly ported engraver2 performs weak due to many nested loops.
- the current engraver.py is a huge script that will get unmaintainable over time if we add more and more futures

# the solution
actually we can better start to move the original engraver.py drawing code to seperate *_drawer.py files element after element. 

the key insight from engraver2: in one loop trough all notes and grace notes we create document_layout.

idea for performance: if we first roughly create a lines list were all events are sorted into lines based on their time and line_break.time
we can break the document_layout loop if we calculated the current page_no to be faster.

'''

# below is a sketch of the document layout data structure. every list of objects contains one example object so the structure is clear.
document_layout = {
    # we split the timeline into lines based on line_break.time. each line contains all events whose time falls within the line's time range.
    'lines': [
        {
            # measured width by getting lowest and highest pitch of all note and grace note events in the line.
            # the lowest/highest pitch x positions define the base line width
            # the beam drawings further extend the line_width so we need to pre measure that as well in the same loop.
            # score.events.line_break.margin_mm is used to trim the layout on the left and right side manually by the user if needed.
            'line_width': 1000.0,
            # the line height is determined by score.layout margin settings and page dimensions + if mini piano is drawn or not in that line.
            'line_height': 100.0,
            # the y top position on the page
            'line_top_y': 0.0,
            '''
                events: we create a draw ready event list containing event types:
                - note
                - continuation_dot
                - stop_symbol
                - arpeggio
                ...
            '''
            'events': [
                    {
                        'type': 'note',
                        'time': 0.0,
                        'event': {
                            # the original event object from score.events.notes or score.events.grace_notes etc...
                        }
                    }
            ]
        }
    ]
}

'''
# the drawing code
after we have created the document layout we can draw line by line without further looping. we gathered all information we need in the document layout already. we just need to loop trough the events of the line.
the drawing code is then split into seperate *_drawer.py files.
'''