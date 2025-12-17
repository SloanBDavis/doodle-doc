from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    modified_time: Mapped[datetime] = mapped_column(DateTime)
    num_pages: Mapped[int] = mapped_column(Integer)


class PageModel(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(36), index=True)
    page_num: Mapped[int] = mapped_column(Integer)
    width_px: Mapped[int] = mapped_column(Integer)
    height_px: Mapped[int] = mapped_column(Integer)
    text_layer: Mapped[str | None] = mapped_column(Text, nullable=True)


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        return self._session_factory()

    def get_document_by_sha256(self, sha256: str) -> DocumentModel | None:
        with self.session() as session:
            return session.query(DocumentModel).filter_by(sha256=sha256).first()

    def add_document(self, doc: DocumentModel) -> None:
        with self.session() as session:
            session.add(doc)
            session.commit()

    def add_page(self, page: PageModel) -> None:
        with self.session() as session:
            session.add(page)
            session.commit()

    def get_document(self, doc_id: str) -> DocumentModel | None:
        with self.session() as session:
            return session.query(DocumentModel).filter_by(doc_id=doc_id).first()

    def get_all_documents(self) -> list[DocumentModel]:
        with self.session() as session:
            return list(session.query(DocumentModel).all())

    def get_pages_for_document(self, doc_id: str) -> list[PageModel]:
        with self.session() as session:
            return list(
                session.query(PageModel)
                .filter_by(doc_id=doc_id)
                .order_by(PageModel.page_num)
                .all()
            )
