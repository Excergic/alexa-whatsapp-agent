# What if content exists but is None/empty?
class Message:
    def __init__(self, content):
        self.content = content

messages = [Message(None), Message(""), Message("hello")]
contents = [m.content for m in messages[-3:]]  # [None, "", "hello"]
result = " ".join(contents)  # 💥 CRASHES! Can't join None