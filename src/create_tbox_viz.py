from graphviz import Digraph
import os

def create_tbox_visualization():
    dot = Digraph(comment='Publication Domain TBOX')
    dot.attr(rankdir='LR')
    
    # Set node styles
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Add classes
    classes = [
        'Paper', 'Author', 'Event', 'Conference', 'Workshop', 'Edition',
        'Journal', 'Volume', 'Proceedings', 'Review', 'Reviewer',
        'Topic', 'Venue'
    ]
    
    for cls in classes:
        dot.node(cls)
    
    # Add subclass relationships
    dot.edge('Conference', 'Event', 'subClassOf')
    dot.edge('Workshop', 'Event', 'subClassOf')
    dot.edge('Reviewer', 'Author', 'subClassOf')
    
    # Add property relationships
    properties = [
        # Paper properties
        ('Paper', 'hasAbstract', 'string'),
        ('Paper', 'hasKeyword', 'Topic'),
        ('Paper', 'cites', 'Paper'),
        ('Paper', 'isPublishedIn', 'Proceedings'),
        ('Paper', 'isPublishedIn', 'Volume'),
        ('Paper', 'hasAuthor', 'Author'),
        ('Paper', 'hasCorrespondingAuthor', 'Author'),
        ('Paper', 'hasReview', 'Review'),
        
        # Author properties
        ('Author', 'hasName', 'string'),
        
        # Conference/Workshop properties
        ('Event', 'hasEdition', 'Edition'),
        
        # Edition properties
        ('Edition', 'hasVenue', 'Venue'),
        ('Edition', 'hasStartDate', 'date'),
        ('Edition', 'hasEndDate', 'date'),
        ('Edition', 'hasYear', 'year'),
        ('Edition', 'hasProceedings', 'Proceedings'),
        
        # Journal properties
        ('Journal', 'hasVolume', 'Volume'),
        
        # Volume properties
        ('Volume', 'hasPublicationDate', 'date'),
        ('Volume', 'hasVolumeNumber', 'integer'),
        
        # Review properties
        ('Review', 'isAssignedBy', 'Reviewer'),
        ('Review', 'hasReviewText', 'string')
    ]
    
    # Add property relationships with different style
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightgreen')
    for source, prop, target in properties:
        prop_node = f"{prop}"
        dot.node(prop_node)
        dot.edge(source, prop_node)
        dot.edge(prop_node, target)
    
    # Add subproperty relationships
    dot.edge('hasCorrespondingAuthor', 'hasAuthor', 'subPropertyOf', style='dashed')
    
    # Create doc/img directory if it doesn't exist
    os.makedirs('doc/img', exist_ok=True)
    
    # Save the visualization to doc/img directory
    dot.render('doc/img/publication_tbox', format='pdf', cleanup=True)

if __name__ == '__main__':
    create_tbox_visualization() 