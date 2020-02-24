class Settings():
    """A class to store the settings"""
    
    def __init__(self):
        """Initializes static settings"""
        self.nrOfIndividuals = 38
        self.individualLength = 2 * self.nrOfIndividuals
        self.volumeCenter = 100
        self.volumeOneTileAway = 50
        self.volumeTwoTilesAway = 25