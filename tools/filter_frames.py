import os
import threading
import queue
from tkinter import Tk, Label, filedialog
from PIL import Image, ImageTk
from collections import deque


PRELOAD_COUNT = 5
UNDO_STACK = deque(maxlen=1)


class ImageCleaner:
    def __init__(self, root, folder):
        print("[INIT] Démarrage du nettoyeur d'images.")
        self.root = root
        self.root.title("Image Cleaner")

        self.folder = folder
        self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"[INIT] {len(self.image_files)} fichiers d'images détectés.")

        self.image_index = 0
        self.preloaded = deque()

        self.queue = queue.Queue()

        self.label = Label(root)
        self.label.pack()

        self.root.bind('a', self.keep)
        self.root.bind('z', self.delete)
        self.root.bind('<Control-z>', self.undo)

        print("[INIT] Démarrage du préchargement.")
        self.load_next_batch_threaded()

    def load_next_batch_threaded(self):
        print("[THREAD] Démarrage d'un thread de préchargement.")

        def loader():
            while len(self.preloaded) < PRELOAD_COUNT and self.image_index < len(self.image_files):
                img_file = self.image_files[self.image_index]
                img_path = os.path.join(self.folder, img_file)
                print(f"[THREAD] Chargement de {img_file}")
                pil_image = Image.open(img_path)
                pil_image.thumbnail((800, 800))
                tk_image = ImageTk.PhotoImage(pil_image)
                self.queue.put((img_file, tk_image))
                self.image_index += 1
            print("[THREAD] Fin de ce batch.")
            self.queue.put(None)

        threading.Thread(target=loader, daemon=True).start()
        self.root.after(100, self.process_queue)

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item is None:
                    print("[QUEUE] Fin du chargement batch.")
                    self.show_image()
                    return
                print(f"[QUEUE] Image préchargée : {item[0]}")
                self.preloaded.append(item)
        except queue.Empty:
            self.root.after(100, self.process_queue)

    def show_image(self):
        if not self.preloaded:
            print("[INFO] Aucune image restante.")
            self.label.config(text="Terminé.")
            return

        self.current_file, self.current_image = self.preloaded[0]
        print(f"[DISPLAY] Affichage de l'image : {self.current_file}")
        self.label.config(image=self.current_image)

    def keep(self, event=None):
        print(f"[KEEP] Image conservée : {self.current_file}")
        self.next_image()

    def delete(self, event=None):
        if self.preloaded:
            filename, _ = self.preloaded.popleft()
            path = os.path.join(self.folder, filename)
            print(f"[DELETE] Suppression de : {filename}")
            if os.path.exists(path):
                os.remove(path)
                UNDO_STACK.append(path)
                print(f"[DELETE] {filename} supprimé du disque.")
            self.load_next_batch_threaded()
            self.show_image()

    def undo(self, event=None):
        if UNDO_STACK:
            path = UNDO_STACK.pop()
            print(f"[UNDO] Impossible d'annuler : '{path}' est déjà supprimé physiquement.")

    def next_image(self):
        if self.preloaded:
            self.preloaded.popleft()
            self.load_next_batch_threaded()
            self.show_image()


def main():
    print("[MAIN] Lancement de l'application.")
    root = Tk()
    folder = filedialog.askdirectory(title="Choisir le dossier d'images")
    if not folder:
        print("[MAIN] Aucun dossier sélectionné.")
        return
    print(f"[MAIN] Dossier sélectionné : {folder}")
    app = ImageCleaner(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
