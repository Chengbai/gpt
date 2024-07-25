# Date: 2024-07-24
# Cheng Bai

from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from typing import List, Union

from utils import load_text_from_file


class BaseTxtDataSource:
    def __init__(sef):
        pass

    @property
    def txt(self):
        raise Exception("Not implemented.")


class FileTxtDataSource(BaseTxtDataSource):
    def __init__(self, file_path: Union[str, Path]):
        super().__init__()

        assert file_path is not None

        self.file_path = Path(file_path)
        self.content = load_text_from_file(self.file_path)

    @property
    def txt(self):
        return self.content

    @staticmethod
    def save(file_path: Union[str, Path], txt):
        with open(file_path, "w") as f:
            f.write(txt)


class DirectoryTxtDataSource(BaseTxtDataSource):
    def __init__(
        self,
        dir_path: Union[str, Path],
        file_patterns: List[str],
        cache_file: str = None,
    ):
        super().__init__()

        assert dir_path is not None
        assert file_patterns

        if cache_file and Path(cache_file).exists():
            self.content = FileTxtDataSource(cache_file).txt
        else:
            self.dir_path = Path(dir_path)
            assert self.dir_path.is_dir()

            self.file_patterns = file_patterns
            pattern_txts = []
            for file_pattern in file_patterns:
                for file_path in tqdm(list(self.dir_path.rglob(file_pattern))):
                    pattern_txts.append(FileTxtDataSource(file_path).txt)

            self.content = "\n".join(pattern_txts)
            cache_file = (
                f"dir_txt_cache_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.txt"
            )
            FileTxtDataSource.save(cache_file, self.content)
            print(f"Cache the director txt into file: {cache_file}")

    @property
    def txt(self):
        return self.content
