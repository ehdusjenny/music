import pretty_midi

#instruments = []
#MIDI note numbers range from 0 - 127
#for note_number in range(128):
#    for instrument in instruments:

        #convert MIDI note numbers to (chord)(accidental)(octave#) String format
#        note_name = pretty_midi.note_number_to_name(note_number)


print(pretty_midi.program_to_instrument_name(128))
