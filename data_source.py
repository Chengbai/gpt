# Date: 2024-07-24
# Cheng Bai

from pathlib import Path
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

        self.dir_path = Path(dir_path)
        assert self.dir_path.is_dir()

        self.file_patterns = file_patterns
        pattern_txts = []
        for file_pattern in file_patterns:
            txt = "\n".join(
                [
                    FileTxtDataSource(file_path).txt
                    for file_path in self.dir_path.rglob(file_pattern)
                ]
            )
            pattern_txts.extend(txt)

        self.content = "\n".join(self.pattern_txts)

    @property
    def txt(self):
        return self.content
